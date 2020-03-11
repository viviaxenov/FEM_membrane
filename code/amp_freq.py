import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import membrane as mb
from config_loader import run_command
import os

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
gamma = 15.
C = gamma*M

omegas = np.linspace(100, 10000, 10000, endpoint=True)
omegas *= 2*np.pi
us = np.zeros_like(f)
respath = "../res/afc/"
os.makedirs(respath, exist_ok=True)
for w in omegas:
    K_eff = K + 1j*w*C - w**2*M
    u0 = sp.linalg.spsolve(K_eff, f)
    grid.a = u0.real.astype(np.float64)
    grid.dump_vtk_grid(os.path.join(respath, f"response{w:.0f}.vtk"))
    us = np.vstack((us, u0))
us = us[1:]
test_nodes = [83, 84, 80, 81]
for n in test_nodes:
    plt.plot(omegas/2/np.pi, np.absolute(us[:, 3*n + 2]), label=f"({grid.x_0[n]:.1f}, {grid.y_0[n]:.1f})")
#    plt.plot(omegas/2/np.pi, us[:, 3*n + 2].real, label=f"Re(f)")
#    plt.plot(omegas/2/np.pi, us[:, 3*n + 2].imag, label=f"Im(f)")
plt.grid()
plt.legend()
plt.xlabel('Frequency, Hz')
plt.ylabel("$\\|f\\|$")
plt.show()
