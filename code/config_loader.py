import membrane as mb
import scipy.sparse
from scipy.sparse.linalg import eigsh
import numpy as np
import pandas as pd

import sympy
from sympy.parsing import sympy_parser

import json
import sys
import os
import pickle
import argparse

from typing import Tuple, List, Union
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence


def initialize_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FEM-membrane --- modelling of membrane deformation and eigenvalue problem",
        epilog="Inspired by BOT-cafe inc.(c) 2019, All rights reserved")

    sp = parser.add_subparsers(dest="mode", help="Possible options; type OPTION --help for info on particular option")

    parser_grid = sp.add_parser("grid", help="Generate and serialize grid from config stored in config file")
    parser_grid.add_argument("config_file", metavar="config_file.json", type=str, help="File with configuration")

    parser_run = sp.add_parser("run", help="Generate or load grid from config file, then perform run")
    parser_run.add_argument("config_file", metavar="config_file.json", type=str, help="File with configuration")

    parser_unpack = sp.add_parser("unpack", help="Unpacks *.json metadata from serialized grid file")
    parser_unpack.add_argument("pickle_path", help="Path to serialized grid file")
    parser_unpack.add_argument("json_path", nargs='?', help="Path where to store unpacked config")

    parser_eigen = sp.add_parser("eigen", help="Find eigenvalues for grid specified in config file")
    parser_eigen.add_argument("config_file", metavar="config_file.json", type=str, help="File with configuration")
    parser_eigen.add_argument("res_dir", metavar="results_directory", type=str, nargs='?',
                              help="Directory to store results")
    parser_eigen.add_argument("--k", type=int, default=1, help="Number of eigenvalues to find")
    parser_eigen.add_argument("--which", choices=['LM', 'SM', 'LI', 'SI', 'LR', 'SR'], default='SM',
                              help="Look for either eigenvalues with small (\'SM\') or large magnitude")
    parser_eigen.add_argument("--tol", type=np.float64, default=0.,
                              help="Precision for solver. Default 0 is machine precision")
    return parser


def parse_elastic_tensor(D: np.ndarray, unit):
    if D is not None:
        D = np.array(D)
        ut = np.triu_indices(6)
        D_buf = np.zeros_like(D)
        D_buf[ut] = D[ut]
        D = D_buf + D_buf.T - np.diag(np.diag(D))
        if 'unit' is not None:
            if unit == "GPa":
                unit = 1e9
            D *= unit
    return D


# checks if an item defines a location (for constraining)
# e.g. is 2-d array [x, y] or one of "top", "bottom", "left", "right", "border"]
def is_loc(item):
    if item in ["top", "bottom", "left", "right", "border"]:
        return True
    try:
        a = np.array(item, dtype=np.float64)
        if a.shape != (2,):
            return False
        return True
    except Exception:
        return False


# checks if an object is a list of locations
# if not, returns the index of first non-compliant element
def is_loc_array(arr):
    if not isinstance(arr, list):
        return False, -1
    for i, item in enumerate(arr):
        if not is_loc(item):
            return False, i
    return True, i


def get_load_expr(expr_string: str):
    x, y, t = sympy.symbols("x, y, t")
    expected_variables = {x, y, t}
    transformations = sympy_parser.standard_transformations + \
                      (sympy_parser.implicit_multiplication_application,)

    expr = sympy_parser.parse_expr(expr_string, local_dict={x: 'x',
                                                            y: 'y',
                                                            t: 't'})
    if len(expr) != 3:
        raise ValueError(f"Load function must specify 3 components, got {len(expr)}")
    for i, comp in enumerate(expr):
        variables = (comp.atoms(sympy.Symbol))
        if not variables <= expected_variables:
            raise ValueError(f"Component {i + 1:d} must depend only on variables x, y, t. Got {variables}")
    return expr


array_of_locs_definition = " e.g. 2-d arrays [x,y] or one of " \
                           "\"top\", \"bottom\", \"left\", \"right\", \"border\""


def check_constraints_dict(constr: dict):
    if 'Displacement' not in constr and 'Velocity' not in constr:
        raise ValueError("Specify one of the sections \"Displacement\" or \"Velocity\" in \"Constraints\" section")
    if 'Displacement' in constr:
        dsp = constr['Displacement']
        res, ind = is_loc_array(dsp)
        if not res:
            msg = "\"Displacements\" must specify an array of locs " + array_of_locs_definition
            if ind >= 0:
                msg += f"Item {ind:d} can't be parsed. Got {dsp[ind]}"
            raise ValueError(msg)

    if 'Velocity' in constr:
        vel = constr['Velocity']
        ok = True
        if not isinstance(vel, list):
            raise ValueError("\"Velocity\" must be a list of structures {\"v\" : [v_x, v_y, v_z], "
                             "\"locs\" : [array of locs]} " + array_of_locs_definition)
        for ind_vel, item in enumerate(vel):
            try:
                a = np.array(item['v'], dtype=np.float64)
                if a.shape != (3,):
                    raise ValueError(f"Velocity must be a 3-d vector, got shape {a.shape}")
            except ValueError as ve:
                raise ValueError(f"\"Velocity\", item {ind_vel + 1}. Failed to parse. Details: {ve}")
            except KeyError as ke:
                raise ValueError(f"\"Velocity\", item {ind_vel + 1}. Missing key \"{ke.args}\"")
            # if only one loc is specified, convert into array of 1
            if is_loc(item['locs']):
                item['locs'] = [item['locs']]
            res, ind_loc = is_loc_array(item['locs'])
            if not res:
                msg = (
                        f"\"Velociy\", entry #{ind_vel} must specify an array of locs " + array_of_locs_definition)
                if ind_loc >= 0:
                    msg += f" but item {ind_loc:d} can't be parsed. Got {item['locs'][ind_loc]}"
                raise ValueError(msg)


def check_grid_dict(grid_dict: dict):
    try:
        rho = grid_dict['rho']
        h = grid_dict['h']
        pickle_file = grid_dict["pickle_file"]
        try:
            geom = grid_dict['Geometric']
        except KeyError:
            raise KeyError("\"Geometric\" section must be specified! Aborting")
        if geom['type'] != 'uniform_rect':
            raise ValueError("Type must be \"uniform_rect\". Aborting\n(PS.: Wait for further releases "
                             "for more sophisticated grid generation routines)")
        all_fields_here = geom['L_x'] and geom['L_y'] and geom['n_y'] and geom['n_x']
        if ('perf_step_x' in geom) != ('perf_step_y' in geom):
            raise ValueError(
                "Both \"perf_step_x\" and \"perf_step_y\" must be specified in order to get perforated grid."
                " Aborting")
        if "Elastic" not in grid_dict:
            raise ValueError("\"Elastic\" section must be specified. Aborting")
        elastic = grid_dict['Elastic']
        if "D" not in elastic and not ("E" in elastic and "nu" in elastic):
            raise ValueError(
                "Either full elastic tensor \"D\" or a pair of elastic constants \"E\" and \"nu\" must be specified.")
        D = elastic.get('D')
        if D is not None:
            try:
                D = np.array(D)
            except ValueError:
                raise ValueError(f"D must be an array of doubles. Got {type(D)}")
            if not D.shape == (6, 6):
                raise ValueError(f"D must be a 6x6 array of doubles. Got {D.shape}")
            unit = elastic.get("unit")
            if unit is not None and not (unit == "GPa" or isinstance(unit, np.float64)):
                raise ValueError("\"unit\" must be either a string \"GPa\" or a number")

        if "Constraints" in grid_dict:
            check_constraints_dict(grid_dict["Constraints"])
        if "damping" in grid_dict:
            damping = grid_dict["damping"]
            keyset = set(damping.keys())
            if not keyset <= {"alpha", "beta"} or len(keyset) == 0:
                raise ValueError("\"Damping\" must specify either one or both keys from \"alpha\", \"beta\"")
    except KeyError as ke:
        print(f"An obligatory key \"{ke.args[0]}\" not found. Aborting")
        raise ke
    except Exception as e:
        raise


def generate_grid_from_dict(grid_dict: dict):
    check_grid_dict(grid_dict)  # ???
    geom = grid_dict['Geometric']
    X = geom['L_x']
    Y = geom['L_y']
    n_x = geom['n_x']
    n_y = geom['n_y']
    rho = grid_dict['rho']
    h = grid_dict['h']
    perforated = ('perf_step_x' in geom) and ('perf_step_y' in geom)
    elastic = grid_dict['Elastic']
    D = parse_elastic_tensor(elastic.get('D'), elastic.get("unit"))

    E = elastic.get('E')
    nu = elastic.get('nu')
    damping = grid_dict.get("damping")
    if perforated:
        step_x = geom['perf_step_x']
        step_y = geom['perf_step_y']
        g = mb.generate_perforated_grid(X, Y, n_x, n_y, step_x, step_y, h, rho, D=D, E=E, nu=nu)
    else:
        g = mb.generate_uniform_grid(X, Y, n_x, n_y, h, rho, E=E, nu=nu, D=D)
    g.ready()
    constr = grid_dict["Constraints"] if "Constraints" in grid_dict else {}
    vel = constr['Velocity'] if "Velocity" in constr else []
    for item in vel:
        it = item['locs']
        loc_array = [it] if is_loc(it) else it
        g.apply_velocity_constraints(loc_array.copy(), item['v'])
    if 'Displacement' in constr:
        dsp = constr['Displacement'].copy()
        g.apply_velocity_constraints(dsp, np.array([0.] * 3))
    g.K = g.K.tocsc()
    g.M = g.M.tocsc()
    if g.damping is not None:
        g.C = g.C.tocsc()
    return g


def check_run_dict(run_dict: dict):
    try:
        all_in = run_dict["vtk_dir"] \
                 and run_dict["n_iter"] \
                 and run_dict["ipf"] \
                 and ("courant" in run_dict or "timestep" in run_dict)
    except KeyError as ke:
        raise KeyError(f"An obligatory key \"{ke.args[0]}\" not found. Aborting")
    if not all_in:
        raise ValueError("Please specify one of keys \"courant\" or \"timestep\"")
    if "solver" in run_dict and run_dict["solver"] not in mb.Grid.supported_solvers:
        slv = run_dict["solver"]
        raise ValueError(f"Solver must be one of {mb.Grid.supported_solvers}; got: {slv}")
    if "Load" in run_dict:
        get_load_expr(run_dict["Load"])


def serialize_grid_and_metadata(pickle_path, g: mb.Grid, metadata: dict):
    """"Function that writes grid together with dict from which it was initialized to pickle file"""
    for e in g.elements:
        e.b = None
    if g.time_solver_type == "factorized":
        g.A = None
    data = {"grid": g, "metadata": metadata}  # packing
    dirname, basename = os.path.split(pickle_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(pickle_path, "wb") as ofile:
        pickle.dump(data, ofile, pickle.HIGHEST_PROTOCOL)


def unpack_pickle(pickle_path) -> Tuple[dict, mb.Grid]:
    with open(pickle_path, "rb") as ifile:
        data = pickle.load(ifile)
    metadata = data["metadata"]
    grid = data["grid"]
    return metadata, grid


def unpack_json_config(pickle_path, json_path=None):
    metadata, _ = unpack_pickle(pickle_path)
    if json_path is None:
        dirname, basename = os.path.split(pickle_path)
        name, _ = os.path.splitext(basename)
        json_path = os.path.join(dirname, name + "_config.json")
    with open(json_path, "w") as ofile:
        json.dump(metadata, ofile, indent=4)


def grid_routine(config_dict: dict):
    grid_dict = config_dict['Grid']
    pickle_path = grid_dict["pickle_file"]
    if not os.path.exists(pickle_path):
        g = generate_grid_from_dict(grid_dict)
        serialize_grid_and_metadata(pickle_path, g, config_dict)
        return g

    metadata, old_grid = unpack_pickle(pickle_path)
    rewrite_needed = False
    if "Grid" not in metadata:
        print(
            f"Metadata in pickled file {pickle_path} is corrupted (\"Grid\" section not specified)."
            f" File will be rewritten")
        rewrite_needed = True
        old_grid_dict = None
    else:
        old_grid_dict = metadata["Grid"]

    damping = grid_dict.pop("damping", None)
    old_grid_dict.pop("damping",
                      None)  # I want to compute damping matrix on every run as I'm changing those params manually
    if grid_dict == old_grid_dict:
        answ: str = input("Grid file is up to date. Still want to rewrite? (Y/n)")
        rewrite_needed = True if answ.lower() in ["y", "yes"] else False
    else:
        rewrite_needed = True
    if damping is not None:
        grid_dict["damping"] = damping
    if rewrite_needed:
        g = generate_grid_from_dict(grid_dict)
        serialize_grid_and_metadata(pickle_path, g, config_dict)
        return g
    else:
        return old_grid


def provide_run_metadata(grid: mb.Grid, config_dict: dict):
    tau = grid.tau
    try:
        n_x = config_dict["Grid"]["Geometric"]["n_x"]
        n_y = config_dict["Grid"]["Geometric"]["n_x"]
    except KeyError as ke:
        raise NotImplementedError("Waring: metadata generation in text format is availiable only for regular grids")
    dmp_dict = {"Thickness [mm]": [1000. * config_dict["Grid"]['h']],
                "Density [$\\frac{g}{cm^3}$]": [config_dict["Grid"]["rho"] / 1000.],
                "Nodes (x)": [n_x],
                "Nodes (y)": [n_y],
                "Perforation frequency (x)": config_dict["Grid"]["Geometric"].get('perf_step_x'),
                "Perforation frequency (y)": config_dict["Grid"]["Geometric"].get('perf_step_y'),
                "Time step [s]": [tau],
                "Iterations": config_dict["Run"].get("n_iter"),
                "Iter. per frame": config_dict["Run"].get("ipf"),
                }
    if "D" not in config_dict["Grid"]["Elastic"]:
        dmp_dict["$E$ [GPa]"] = config_dict["Grid"]["Elastic"]["E"] / 1e9
        dmp_dict["$\\nu$"] = config_dict["Grid"]["Elastic"]["nu"]
    else:
        inds = [(i, j) for i in range(0, 6) for j in range(0, 6) if i <= j]
        unit = config_dict["Grid"]["Elastic"].get("unit")
        for idx in inds:
            val = config_dict["Grid"]["Elastic"]["D"][idx[0]][idx[1]]
            if val == 0.:
                continue
            key = f"$c_{{{idx[0] + 1}{idx[1] + 1}}}$"
            if unit == "GPa":
                key += "[GPa]"
            dmp_dict[key] = [val]
    data = pd.DataFrame.from_dict(dmp_dict)
    data_path = os.path.join(config_dict["Run"]["vtk_dir"], "details.tex")
    with open(data_path, "w") as ofile:
        data.T.to_latex(ofile, escape=False)


def run_routine(config_dict: dict, grid: mb.Grid):
    run_dict = config_dict["Run"]
    if "Load" in run_dict:
        load_expr = get_load_expr(run_dict["Load"])
        grid.set_loading(load_expr)
    else:
        for e in grid.elements:
            e.b = lambda t: np.zeros(3)  # TODO: fix this bullshit
    timestep = run_dict['courant'] * grid.estimate_tau() if 'courant' in run_dict else run_dict["timestep"]
    grid.set_time_integration_mode(timestep, run_dict.get("solver"))

    try:
        provide_run_metadata(grid, config_dict)
    except NotImplementedError as e:
        print(str(e))

    if "damping" in config_dict["Grid"]:
        damping = run_dict["damping"]
        grid.set_rayleigh_damping(**damping)
    n_iter = run_dict['n_iter']
    ipf = run_dict['ipf']
    vtk_dir = run_dict['vtk_dir']

    if not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir)
    grid.dump_vtk_grid(os.path.join(vtk_dir, "f0"))

    for i in range(n_iter):
        grid.iteration()
        if (i + 1) % ipf == 0:
            grid.dump_vtk_grid(os.path.join(vtk_dir, f"f{(i + 1)//ipf:d}"))


def get_config_dict(config_path):
    with open(config_path, mode='r') as infile:
        lines_without_comments = [line for line in infile.readlines() if not line.lstrip().startswith("#")]
        # as json doesn't support comments by standard, we have to strip them manually
    try:
        loaded: dict = json.loads(''.join(lines_without_comments), parse_float=np.float64)
        return loaded
    except (json.JSONDecodeError, json.decoder.JSONDecodeError) as parse_error:
        raise ValueError(f"Failed to parse config file.\n"
                         f"In line {parse_error.lineno} symbol {parse_error.colno}: {parse_error.msg}")


def eigen_routine(grid: mb.Grid, args_dict: dict):
    if grid.damping is None:
        M_eff = scipy.sparse.csc_matrix(grid.M)
        A_eff = scipy.sparse.csc_matrix(grid.K)
        try:
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(A_eff, M=M_eff, **args_dict)  # symmeric matrices
        except scipy.sparse.linalg.ArpackNoConvergence as e:
            print(str(e))
            eigvals = e.eigenvalues
            eigvecs = e.eigenvectors
        freqs = np.sqrt(np.abs(eigvals)) / 2. / np.pi
        return {"eigenvalues": eigvals, "frequencies": freqs}, eigvecs
    else:
        # [[ M, 0],             [[C, K],
        #   [0, -M]] * alpha +   [M, 0]]
        zero = scipy.sparse.csc_matrix((grid.M.shape[0], grid.M.shape[0]))
        M_eff = scipy.sparse.vstack(
            (scipy.sparse.hstack((-grid.M, zero)),
             scipy.sparse.hstack((zero, grid.M))), format="csc")
        A_eff = scipy.sparse.vstack(
            (scipy.sparse.hstack((grid.C, grid.K)),
             scipy.sparse.hstack((grid.M, zero))), format="csc")
        try:
            eigvals, eigvecs = scipy.sparse.linalg.eigs(A_eff, M=M_eff, **args_dict)
        except scipy.sparse.linalg.ArpackNoConvergence as e:
            print(str(e))
            eigvals = e.eigenvalues
            eigvecs = e.eigenvectors
        freqs = np.imag(eigvals) / 2. / np.pi
        damping = np.real(eigvals).astype(np.float64)
        return {"eigenvalues": eigvals, "frequencies": freqs, "damping": damping}, eigvecs


def run_command(command: Union[str, List[str]]):
    if isinstance(command, str):
        command = command.split()
    parser = initialize_argparser()
    parsed_args = parser.parse_args(command)

    mode = parsed_args.mode

    if mode == 'unpack':
        unpack_json_config(parsed_args.pickle_path, parsed_args.json_path)
    elif mode == 'grid':
        config_dict = get_config_dict(parsed_args.config_file)
        grid_routine(config_dict)
    elif mode == 'run':
        config_dict = get_config_dict(parsed_args.config_file)
        if "Run" not in config_dict:
            print("Specify \"Run\" section in the config file if you want to perform runs")
            exit()
        run_dict = config_dict["Run"]
        check_run_dict(run_dict)
        grid = grid_routine(config_dict)
        run_routine(config_dict, grid)
    elif mode == 'eigen':
        args_dict = vars(parsed_args)
        args_dict.pop("mode")
        config_file = args_dict.pop("config_file")
        res_dir = args_dict.pop("res_dir")
        try:
            config_dict = get_config_dict(config_file)
        except ValueError as ve:
            print(str(ve))
            exit(1)
        grid = grid_routine(config_dict)
        if "damping" in config_dict["Grid"]:
            damping = config_dict["Grid"]["damping"]
            grid.set_rayleigh_damping(**damping)
        res_dict, eigvecs = eigen_routine(grid, args_dict)

        dir, name = os.path.split(config_file)
        fname, ext = os.path.splitext(name)
        name = fname + "_" + ext[1:]
        if res_dir is None:
            res_dir = os.path.join(dir, "eigen_results", name)
        else:
            res_dir = os.path.join(res_dir, name, "eigen_results")
        os.makedirs(res_dir, exist_ok=True)
        file = os.path.join(res_dir, "eigenvectors")
        np.save(file, eigvecs)
        file = os.path.join(res_dir, "eigvalues")
        np.savez(file, **res_dict)
        arr = np.array([res_dict[key] for key in res_dict.keys()]).T
        names = ", ".join(res_dict.keys())
        file = os.path.join(res_dir, "res.csv")
        np.savetxt(file, arr, delimiter=', ',
                   header=names)  # , fmt=['(%+1.3e, %+1.3e1j)', '%1.5e+%.1ej', '%1.5e+%.1ej'])

        res_dict["eigvecs"] = eigvecs
        res_dict["grid"] = grid
        res_dict["config"] = config_dict
        return res_dict


if __name__ == "__main__":
    run_command(sys.argv[1:])
