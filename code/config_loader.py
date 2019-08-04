import code.membrane as mb
import numpy as np

import sympy
from sympy.parsing import sympy_parser

import json
import sys, os
import pickle


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


def check_load_expr(expr_string: str):
    x, y, t = sympy.symbols("x, y, t")
    expected_variables = {x, y, t}
    transformations = sympy_parser.standard_transformations + \
                      (sympy_parser.implicit_multiplication_application,)

    expr = sympy_parser.parse_expr(expr_string, local_dict={x: 'x',
                                                            y: 'y',
                                                            t: 't'})
    if len(expr) != 3:
        return False, f"Load function must specify 3 components, got {len(expr)}"
    for i, comp in enumerate(expr):
        variables = (comp.atoms(sympy.Symbol))
    if not variables <= expected_variables:
        return False, f"Component {i + 1:d} must depend only on variables x, y, t. Got {variables}"
    return True, expr


array_of_locs_definition = " e.g. 2-d arrays [x,y] or one of " \
                           "\"top\", \"bottom\", \"left\", \"right\", \"border\""


def run_from_config(config_path: str):
    loaded = []
    serialization_needed = True
    with open(config_path, mode='r') as infile:
        lines_without_comments = [line for line in infile.readlines() if not line.lstrip().startswith("#")]
        # as json doesn't support comments by standard, we have to strip them manually
        try:
            loaded: dict = json.loads(''.join(lines_without_comments), parse_float=np.float64)
        except json.JSONDecodeError as parse_error:
            print("Failed to parse config file. Details: " + parse_error.msg)
            exit(-1)
        print("Parsing config")
        run_needed = 'Run' in loaded
        try:
            rho = loaded['rho']
            h = loaded['h']
            pickle_file = loaded["pickle_file"]
        except KeyError as ke:
            print(f"An obligatory key \"{ke.args[0]}\" is missing, please, specify in the config. Aborting")

        if os.path.exists(pickle_file):
            pickle_modified = os.path.getmtime(pickle_file)
            config_modified = os.path.getmtime(config_path)
            if pickle_modified >= config_modified:
                answ = input("Serialized grid file is up to date. Still want to rewrite? (Y/n)")
                if answ.lower() in ['y', 'yes']:
                    serialization_needed = True
                else:
                    serialization_needed = False
        else:
            serialization_needed = True

        if not (serialization_needed or run_needed):
            exit()

        if serialization_needed:
            geom: dict
            try:
                geom = loaded['Geometric']
            except KeyError:
                print("\"Geometric\" section must be specified! Aborting")
                exit(-1)
            try:
                if geom['type'] != 'uniform_rect':
                    print("Type must be \"uniform_rect\". Aborting\n(PS.: Wait for further releases "
                          "for more sophisticated grid generation routines)")
                    exit(-1)
                X = geom['L_x']
                Y = geom['L_y']
                n_x = geom['n_x']
                n_y = geom['n_y']
            except KeyError as ke:
                print(f"An obligatory key \"{ke.args[0]}\" in \"Geometric\" section not found. Aborting")
                exit(-1)

            perforated = False
            step_x = None
            step_y = None
            if ('perf_step_x' in geom) != ('perf_step_y' in geom):
                print("Both \"perf_step_x\" and \"perf_step_y\" must be specified in order to get perforated grid."
                      " Aborting")
                exit(-1)
            else:
                perforated = ('perf_step_x' in geom) and ('perf_step_y' in geom)
                if perforated:
                    step_x = geom['perf_step_x']
                    step_y = geom['perf_step_y']

            elastic: dict
            try:
                elastic = loaded['Elastic']
            except KeyError as ke:
                print("\"Elastic\" section must be specified. Aborting")
                exit(-1)

            D = elastic.get('D')
            if D is not None:
                try:
                    D = np.array(D)
                except ValueError:
                    print(f"D must be an array of doubles. Got {type(D)}")
                    exit(-1)
                if not D.shape == (6, 6):
                    print(f"D must be a 6x6 array of doubles. Got {D.shape}")
                    exit(-1)
                ut = np.triu_indices(6)
                D_buf = np.zeros_like(D)
                D_buf[ut] = D[ut]
                D = D_buf + D_buf.T - np.diag(np.diag(D))
                if 'unit' in elastic:
                    unit = elastic['unit']
                    if unit == "GPa":
                        unit = 1e9
                    elif not isinstance(unit, np.float64):
                        # TODO: more units
                        print("\"unit\" must be either a string \"GPa\" or a number")
                        exit(-1)
                    D *= unit

            E = elastic.get('E')
            nu = elastic.get('nu')

            if 'Constraints' in loaded:
                constr = loaded['Constraints']
                if 'Displacement' not in constr and 'Velocity' not in constr:
                    print("Specify one of the sections \"Displacement\" or \"Velocity\" in \"Constraints\" section")
                    exit(-1)

                if 'Displacement' in constr:
                    dsp = constr['Displacement']
                    res, ind = is_loc_array(dsp)
                    if not res:
                        print("\"Displacements\" must specify an array of locs " + array_of_locs_definition)
                        if ind >= 0:
                            print(f"Item {ind:d} can't be parsed. Got {dsp[ind]}")
                        exit(-1)

                if 'Velocity' in constr:
                    vel = constr['Velocity']
                    ok = True
                    if not isinstance(vel, list):
                        print("\"Velocity\" must be a list of structures {\"v\" : [v_x, v_y, v_z], " \
                              "\"locs\" : [array of locs]} " + array_of_locs_definition)
                        exit(-1)

                    for ind_vel, item in enumerate(vel):
                        try:
                            a = np.array(item['v'], dtype=np.float64)
                            if a.shape != (3,):
                                raise ValueError(f"Velocity must be a 3-d vector, got shape {a.shape}")
                        except ValueError as ve:
                            print(f"\"Velocity\", item {ind_vel + 1}. Failed to parse. Details: {ve}")
                        except KeyError as ke:
                            print(f"\"Velocity\", item {ind_vel + 1}. Missing key \"{ve.args}\"")
                        # if only one loc is specified, convert into array of 1
                        if is_loc(item['locs']):
                            item['locs'] = [item['locs']]
                        res, ind_loc = is_loc_array(item['locs'])
                        if not res:
                            print(
                                f"\"Velociy\", entry #{ind_vel} must specify an array of locs " + array_of_locs_definition)
                            if ind >= 0:
                                print(f"but item {ind:d} can't be parsed. Got {item['locs'][ind]}")
                            exit(-1)
        run_params = []
        if 'Run' in loaded:
            run_params = loaded['Run']
            try:
                all_fine = (True
                            and isinstance(run_params['vtk_dir'], str)
                            and isinstance(run_params['n_iter'], int)
                            and ('timestep' in run_params != 'courant' in run_params)
                            )
            except KeyError as ke:
                print(f"An obligatory key \"{ke.args[0]}\" in \"Elastic\" section not found. Aborting")
                exit(-1)
            except Exception:
                raise

        load_expr = None
        if "Load" in loaded:
            try:
                parsed = check_load_expr(loaded["Load"])
                if parsed[0]:
                    load_expr = parsed[1]
                else:
                    print(parsed[1])
                    exit(-1)
            except Exception as e:
                print("Error while parsing Load function")
                exit(-1)

    pickle_dir = os.path.dirname(pickle_file)
    try:
        os.makedirs(pickle_dir)
    except Exception as e:
        if not isinstance(e, FileExistsError):
            raise
    if run_needed:
        try:
            os.makedirs(run_params['vtk_dir'])
        except Exception as e:
            if not isinstance(e, FileExistsError):
                raise
    g = []
    if not serialization_needed:
        print(f"Loading serialized grid from {pickle_file}")
        with open(pickle_file, 'rb') as ifile:
            g = pickle.load(ifile)
        print("Done")
    else:
        print("Generating grid")
        if perforated:
            g = mb.generate_perforated_grid(X, Y, n_x, n_y, step_x, step_y, h, rho, D=D, E=E, nu=nu)
        else:
            g = mb.generate_uniform_grid(X, Y, n_x, n_y, h, rho, E=E, nu=nu, D=D)
        print("Done")

        g.ready()
        constr = []
        if 'Constraints' in loaded:
            constr = loaded['Constraints']
        if 'Velocity' in constr:
            vel = constr['Velocity']
            for item in vel:
                it = item['locs']
                loc_array = [it] if is_loc(it) else it
                g.apply_velocity_constraints(loc_array, item['v'])
        if 'Displacement' in constr:
            dsp = constr['Displacement']
            g.apply_velocity_constraints(dsp, np.array([0.] * 3))
        g.K = g.K.tocsc()
        g.M = g.M.tocsc()

        print("Serializing grid")
        g.serialize(pickle_file)
        print(f"Serialized grid stored at {pickle_file}")

    if run_needed:
        if load_expr is not None:
            g.set_loading(load_expr)
        else:
            for e in g.elements:
                e.b = lambda t: np.zeros(3)
        new_timestep = 0.
        if "courant" in run_params:
            new_timestep = run_params['courant'] * g.estimate_tau()
        else:
            new_timestep = run_params['timestep']

        print("Preparing for time integration")
        if 'solver' not in run_params:
            g.set_time_integration_mode(new_timestep)
        else:
            g.set_time_integration_mode(new_timestep, run_params['solver'])
        print("Done")

        n_iter = run_params['n_iter']
        ipf = run_params['ipf']
        vtk_dir = run_params['vtk_dir']

        g.dump_vtk_grid(vtk_dir + "f0")

        print("Performig iterations")
        for i in range(n_iter):
            g.iteration()
            if (i + 1) % ipf == 0:
                g.dump_vtk_grid(vtk_dir + f"f{(i + 1)//ipf:d}")
        print("Done")
        print(f".vtk dumps stored at {vtk_dir}")


if __name__ == "__main__":
    import sys

    sys.argv.append("../configs/anisotropic.json")
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        run_from_config(config_path)
    else:
        print(f"USAGE: {os.path.basename(sys.argv[0])} config_file.json")
        exit()
