import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

import code.membrane as mb
from code.config_loader import run_command

import os
import pandas as pd


# these are analytical frequencies for vertical movement
# that is determined by equation
# a^2 w_tt = w_xx + w_yy
# a^2 = G/\rho --- speed of this type of waves, G = E/2(1 + nu)
# (movement in "plain stress component" is trickier)
def get_analytical_freqs(K, M, Lx, Ly, a):
    freqs = [[0.5 * a * np.sqrt(k ** 2 / Lx ** 2 + m ** 2 / Ly ** 2) for k in range(1, K + 1)] for m in
             range(1, M + 1)]
    return np.array(freqs)


res_dict = run_command("eigen ../configs/isotropic.json --k 30 --which SM --tol 1e-11")
os.makedirs("../res/modes/", exist_ok=True)
np.savetxt("../res/modes/raw_eigvecs.csv", res_dict["eigvecs"], delimiter=',')
np.savetxt("../res/modes/raw_eigvals.csv", res_dict["eigenvalues"], delimiter=',')

g: mb.Grid = res_dict["grid"]
N = g.n_nodes

# multiplying vector(s) by this matrix returns their vertical components
# columns form the basis of vertical modes subspace
vertical_subspace = sp.dok_matrix((3 * N, N))
for i in range(N):
    vertical_subspace[3 * i + 2, i] = 1.
vertical_subspace = vertical_subspace.tocsr()

projection_tolerance = 1e-11  # it's actually solver's tolerance
ev = res_dict["eigvecs"].T  # rows of this are eigenvectors
norms = np.linalg.norm(ev, ord=np.inf, axis=1)

for i in range(ev.shape[1]):
    ev[:, i] /= norms

vertical_components = ev @ vertical_subspace

vertical_idx = np.linalg.norm(vertical_components, ord=np.inf, axis=1) > 1 - projection_tolerance
vertical_freqs = res_dict["frequencies"][vertical_idx]
vertical_modes = ev[vertical_idx]

for i in range(vertical_modes.shape[0]):
    g.a = vertical_modes[i] * 0.1
    g.set_sigma()
    g.dump_vtk_grid(f"../res/modes/mode{i}.vtk")

E = 1e6
nu = 0.45
rho = 900.
a = np.sqrt(E / 2. / (1. + nu) / rho)
analytical_freqs = get_analytical_freqs(6, 6, 1., 2., a).flatten()
analytical_freqs.sort()
analytical_freqs = analytical_freqs[:vertical_freqs.shape[0]]

df = pd.DataFrame({"Analytical": analytical_freqs,
                   "Numeric": vertical_freqs,
                   "Rel. err., %": np.abs(analytical_freqs - vertical_freqs) / vertical_freqs * 100})
with open("../res/modes/frequencies.html", "w") as ofile:
    df.to_html(ofile)
with open("../res/modes/frequencies.tex", "w") as ofile:
    df.to_latex(ofile)

