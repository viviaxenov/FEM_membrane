import membrane as mb

import numpy as np

def get_node_index(node_cords : np.ndarray, n_x : int):
    return node_cords[0] + (n_x + 1)*node_cords[1]                    #

E = 1e6
nu = 0.45
h = 0.001
rho = 900
magnitude = 1e8
spd = 1


n_0 = input('n_0: ')

g = mb.generate_uniform_grid(size, size, n_0, n_0, E, nu, h, rho)       # created a grid with given params
g.ready()                                                               # preparing grid
g.constrain_velocity(get_node_index([20, 20], 40), [0.0, 0.0, 1.0]*spd)
g.K = g.K.tocsc()
g.M = g.M.tocsc()
g.a_tt = sp.linalg.spsolve(g.M, -(g.K.dot(g.a) + g.f))            # count acceleration to satisfy equation

tau = g.estimate_tau()

g.set_Newmark_noinverse(0.5, 0.5, tau)

n_iter = input('Number of iterations: ')

g.dump_vtk_grid("../res/constraints/constr_v")
for i in range(n_iter):
	g.iteration_Newmark_noinverse()
	g.dump_vtk_grid("../res/constraints/constr_v{0:d}".format(i + 1))
		
