import numpy as np
import code.membrane as mb

from code.utils import *

import os
import pickle

E = 1e6
nu = 0.45
h = 0.001
rho = 900
size_x = 0.2
f_mag = 3000000.0
v_mag = 20.0
theta = np.pi/6

n_steps = 50
perf_step = 5

n_iter = 50

iter_strike = 25
iter_after = 75

ipf = 16

# grid_path = f'../pickles/fix_side_perf{n_steps}x{n_steps}_p{perf_step}x{perf_step}'
grid_path = f'../pickles/fix_side{n_steps}x{n_steps}'
dmp_path = f'../res/fix_side/'
if not os.path.exists(dmp_path):
    os.mkdir(dmp_path)

g: mb.Grid

if not os.path.exists(grid_path):
    g = mb.generate_uniform_grid(size_x, size_x, n_steps, n_steps, E, nu, h, rho)

    for i in range(g.x_0.shape[0]):
        if g.y_0[i] == size_x:
            g.constrain_velocity(i, np.zeros(3))

    g.ready()

    g.K = g.K.tocsc()
    g.M = g.M.tocsc()
    tau = g.estimate_tau()
    g.set_Newmark_params(0.5, 0.5, tau)
    with open(grid_path, 'wb') as ofile:
        pickle.dump(g, ofile, protocol=pickle.HIGHEST_PROTOCOL)

with open(grid_path, 'rb') as ifile:
    g = pickle.load(ifile)

for i in range(2*n_steps):
    g.elements[i].b = np.array([0.0, 0.0, 1.0])*f_mag
g.assemble_f()

g.dump_vtk_grid(dmp_path + 'f0')
for i in range(1, iter_strike + 1):
    for _ in range(ipf):
        g.iteration_Newmark()
    g.dump_vtk_grid(dmp_path + f'f{i}')

for i in range(2*n_steps):
    g.elements[i].b = np.array([0.0, 0.0, 0.0])*f_mag
g.assemble_f()

for i in range(1 + iter_strike, iter_after + iter_strike + 1):
    for _ in range(ipf):
        g.iteration_Newmark()
    g.dump_vtk_grid(dmp_path + f'f{i}')


