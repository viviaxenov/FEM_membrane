import code.membrane as mb
from code.utility import get_node_index

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

import matplotlib.pyplot as plt


E = 1e6
size = 1.0
nu = 0.45
h = 0.001
rho = 900
magnitude = 1e8
spd = 50

n_0 = int(input('n_0: '))

strike_point = np.array([n_0 // 2, n_0 // 2], dtype=int)


g = mb.generate_uniform_grid(size, size, n_0, n_0, E, nu, h, rho)  # created a grid with given params
g.ready()  # preparing grid
strike_ind = (get_node_index(strike_point,  n_0))

g.constrain_velocity(strike_ind, np.array([0.0, 0.0, 1.0]) * spd)
g.K = g.K.tocsc()
g.M = g.M.tocsc()
g.a_tt = sp.linalg.spsolve(g.M, -(g.K.dot(g.a) + g.f))  # count acceleration to satisfy equation

tau = g.estimate_tau() / 100

g.set_Newmark_noinverse(0.5, 0.5, tau)

n_shots = int(input("Number of frames: "))
n_iter = int(input('Number of iterations per frame: '))

g.dump_vtk_grid("../res/constraints/constr_v0")
for i in range(n_shots):
    for j in range(n_iter):
        g.iteration_Newmark_noinverse()
    g.dump_vtk_grid("../res/constraints/constr_v{0:d}".format(i + 1))

strike_pos = np.array([g.x_0[strike_ind], g.y_0[strike_ind]])

r = []
w = []
v_mag = []

for ind in range(g.n_nodes):
    curr_pos = np.array([g.x_0[ind], g.y_0[ind]])
    r_c = np.linalg.norm(curr_pos - strike_pos, ord=2)
    r += [r_c]
    w += [g.a[3*ind + 2]]
    vel = g.a_t[3*ind : 3*(ind + 1)]
    v_mag += [np.linalg.norm(vel)]

r = np.array(r)
w = np.array(w)
v_mag = np.array(v_mag)

fig, axs = plt.subplots(1, 2, sharex=True)
ax = axs[0]
ax.plot(r, w, 'bs')
ax.grid(True)
ax = axs[1]
ax.plot(r, v_mag, 'rs')
ax.grid(True)
plt.show()






