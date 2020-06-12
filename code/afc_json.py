import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import membrane as mb
from config_loader import run_command
import os
import sys
import time
import json
import pandas as pd

def assemble_matrices(json_task_path):

    grid = run_command("grid " + json_task_path)

    def adhoc_assemble_f():
            grid.f.fill(0.0)
            for ind in grid.element_groups['load']:
                elem = grid.elements[ind]
                f_elem = elem.S * elem.h * 1e8*np.array([0.,0.,1.]) / 6.0
                for i in range(3):
                    I = elem.node_ind[i]
                    grid.f[3*I:3*(I + 1)] -= f_elem
    grid.assemble_f = adhoc_assemble_f
    grid.assemble_f()

    K, M, f = grid.K, grid.M, grid.f
    return grid, K, M, f

def parse_json_task(json_task_path):
    grid, K, M, f = assemble_matrices(json_task_path)

    with open(json_task_path, "r") as ifile:
        lines = [l for l in ifile.readlines() if l[0] != '#']
    s = "".join(lines)
    dct = json.loads(s)
    afc_config = dct['AFC']

    freqs = afc_config['freqs_Hz']
    freqs = np.linspace(freqs[0], freqs[1], int(freqs[2]) + 1, endpoint=True)
    cp_idx = afc_config['cp_idx']
    gammas = afc_config['gammas']

    return grid, K, M, f, freqs, cp_idx, gammas


def get_afc(K, M, f, gamma, omegas):
    n_freqs = omegas.shape[0]
    us = np.zeros((n_freqs, f.shape[0]), dtype=np.complex128)
    C = gamma*M
    for i in range(n_freqs):
        w = omegas[i]
        K_eff = K + 1j*w*C - w**2*M
        u0, res = sp.linalg.bicgstab(K_eff, f)
        us[i, :] = u0
    return us


if __name__ == '__main__':

    if not len(sys.argv) > 1:
        print(f"Usage: {sys.argv[0]} task.json")
        exit(-1)

    json_task_path = sys.argv[1]
    task_name = os.path.splitext(os.path.basename(json_task_path))[0]
    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    common_path_prefix = os.path.join('../res/cmcm2020_afc/', task_name, timestamp)
    os.makedirs(common_path_prefix, exist_ok=True)

    print(f"Assembling matrices for task {task_name}")
    start = time.perf_counter()
    grid, K, M, f, freqs, test_nodes, gammas = parse_json_task(json_task_path)
    t_elapsed = time.perf_counter() - start
    print(f"Assembly done! Elapsed time: {t_elapsed:f}")

    omegas = 2*np.pi*freqs
    print(f"Building AFC for {len(gammas)} damping parameters at {len(freqs)} frequencies")
    
    res_dict = {}

    for gamma in gammas:
    
        gamma_str = "gamma_" + str(gamma).replace(".", "_")
        output_path_prefix = os.path.join(common_path_prefix, gamma_str)
        print(f"Evaluating AFC for gamma = {gamma:.2e}")
        start = time.perf_counter()
        us = get_afc(K, M, f, gamma, omegas)
        t_elapsed = time.perf_counter() - start
        print(f"Elapsed time: {t_elapsed:f}")

        os.makedirs(os.path.join(output_path_prefix, "modes"), exist_ok=True)

        for idx, freq in enumerate(freqs):
            f_str = f"freq_{freq:.4f}".replace('.', '_')
            output_path = os.path.join(output_path_prefix, "modes", "mode_" + f_str)
            grid.a = us[idx, :].real
            grid.a_t[:] = 0.
            grid.a_tt[:] = 0.
            grid.dump_vtk_grid(output_path)

        for n in test_nodes:
            u = us[:, 3*n + 2]
            res_dict[(gamma, n)] = u.copy()
            coords = (grid.x_0[n], grid.y_0[n])
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs.plot(omegas/2/np.pi, np.absolute(us[:, 3*n + 2]), label='$\\|u(\\nu)\\|$)')
            axs.plot(omegas/2/np.pi, us[:, 3*n + 2].real, label=f"$Re u(\\nu)$")
            axs.plot(omegas/2/np.pi, us[:, 3*n + 2].imag, label=f"$Im u(\\nu)$")

            axs.grid()
            axs.legend()
            axs.set_xlabel('Frequency, Hz')

            fig.suptitle(f"AFC for control point ${coords}, \\gamma = {gamma:.2f}$")
            fig.savefig(os.path.join(output_path_prefix, f"cp_{n:d}.pdf"), fmt='pdf')

    ds = pd.DataFrame(res_dict, index=omegas)
    ds.columns = pd.MultiIndex.from_tuples(ds.columns, names=["gamma", "cp"])
    ds.to_hdf(os.path.join(common_path_prefix, "data.h5"), "w")
    ds.to_pickle(os.path.join(common_path_prefix, "data.pkl"))
    print(ds)
    print("Done")
