import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import numba as nb

import membrane as mb
from config_loader import run_command
import os
import sys
import time

def reassemble_matrices():

    grid = run_command("grid ../configs/groups_test.json")

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
    os.makedirs('matrices/', exist_ok=True)
    sp.save_npz('matrices/K.npz', K)
    sp.save_npz('matrices/M.npz',  M)
    np.savez('matrices/f.npz', f=f)
    return K, M, f

def get_afc_dumb(K, M, f, gamma, omegas):
    n_freqs = omegas.shape[0]
    us = np.zeros((n_freqs, f.shape[0]), dtype=np.complex128)
    C = gamma*M
    for i in range(n_freqs):
        w = omegas[i]
        K_eff = K + 1j*w*C - w**2*M
        #u0, res = sp.linalg.bicgstab(K_eff, f)
        u0 = np.linalg.solve(K_eff, f)
        us[i, :] = u0
    return us

@nb.njit('c16[:,:](f8[:,:],f8[:,:],f8[:],f8,f8[:])', parallel=True, cache=True)
def get_afc(K, M, f, gamma, omegas):
    n_freqs = omegas.shape[0]
    us = np.zeros((n_freqs, f.shape[0]), dtype=np.complex128)
    C = gamma*M
    f = f.astype(np.complex128)
    for i in nb.prange(n_freqs):
        w = omegas[i]
        K_eff = K + 1j*w*C - w**2*M
        u0 = np.linalg.solve(K_eff, f)
        us[i, :] = u0
    return us

if __name__ == '__main__':

    if '-r' in sys.argv or not os.path.exists('matrices/'):
        K, M, f = reassemble_matrices()
    else:
        K = sp.load_npz('matrices/K.npz')
        M = sp.load_npz('matrices/M.npz')
        f = np.load('matrices/f.npz')['f']

    gamma = 15.

#    n_freqs = sys.argv[1] if len(sys.argv) > 1 else 1000
    n_freqs = 100

    omegas = np.linspace(100, 5000, n_freqs, endpoint=True)
    omegas *= 2*np.pi
    respath = "../res/afc/"
    os.makedirs(respath, exist_ok=True)

    K = K.todense()
    M = M.todense()

    nb.config.NUMBA_NUM_THREADS = 4
    start = time.perf_counter()
    us1 = get_afc(K, M, f, gamma, omegas)
    print(f"Optimized: {(time.perf_counter() - start):f}")

    start = time.perf_counter()
    us = get_afc_dumb(K, M, f, gamma, omegas)
    print(f"No optimization: {(time.perf_counter() - start):f}")

    assert np.allclose(us, us1, rtol=1e-9, atol=1e-16)

    test_nodes = [83, 84, 80, 81]
    for n in test_nodes:
        plt.plot(omegas/2/np.pi, np.absolute(us[:, 3*n + 2]), )#label=f"({grid.x_0[n]:.1f}, {grid.y_0[n]:.1f})")
    #    plt.plot(omegas/2/np.pi, us[:, 3*n + 2].real, label=f"Re(f)")
    #    plt.plot(omegas/2/np.pi, us[:, 3*n + 2].imag, label=f"Im(f)")
    plt.grid()
    plt.legend()
    plt.xlabel('Frequency, Hz')
    plt.ylabel("$\\|f\\|$")
    plt.show()
