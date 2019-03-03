import code.membrane as mb

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

from code.utility import get_node_index

E = 1e6
size = 1.0
nu = 0.45
h = 0.001
rho = 900
magnitude = 1e8
spd = 1e2


n_0 = int(input('n_0: '))

g = mb.generate_uniform_grid(size, size, n_0, n_0, E, nu, h, rho)       # created a grid with given params
g.ready()                                                               # preparing grid
ind = (get_node_index(np.array([n_0//2, n_0//2], dtype=int), n_0))

g.constrain_velocity(ind, np.array([0.0, 0.0, 1.0])*spd)
g.K = g.K.tocsc()
g.M = g.M.tocsc()
g.a_tt = sp.linalg.spsolve(g.M, -(g.K.dot(g.a) + g.f))            # count acceleration to satisfy equation

tau = g.estimate_tau()/100

g.set_Newmark_noinverse(0.5, 0.5, tau)

n_shots = int(input("Number of frames: "))
n_iter = int(input('Number of iterations per frame : '))

g.dump_vtk_grid("../res/constraints/constr_v")
for i in range(n_shots):
    for j in range(n_iter):
        g.iteration_Newmark_noinverse()
    g.dump_vtk_grid("../res/constraints/constr_v{0:d}".format(i + 1))
		
