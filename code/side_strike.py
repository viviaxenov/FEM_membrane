import numpy as np
import code.membrane as mb
import os
import pickle

import configparser as cfp

from code.utils import *

parser = cfp.ConfigParser()
parser.read('../configs/side_strike.ini')

args_dict = parser['DEFAULT']

E = np.float64(args_dict['E'])
nu = np.float64(args_dict['nu'])
h = np.float64(args_dict['h'])
rho = np.float64(args_dict['rho'])
size_x = np.float64(args_dict['size_x'])
f_mag = np.float64(args_dict['f_mag'])
n_steps = int(args_dict['n_steps'])
iters_strike = int(args_dict['iters_strike'])
iters_after = int(args_dict['iters_after'])
ipf = int(args_dict['ipf'])

grid_path = f'../pickles/ug{n_steps}x{n_steps}'
dmp_path = f'../res/wconfig/side_strike/'
if not os.path.exists(dmp_path):
    os.makedirs(dmp_path)

g : mb.Grid

if not os.path.exists(grid_path):
    g = mb.generate_uniform_grid(size_x, size_x, n_steps, n_steps, E, nu, h, rho)
    g.ready()
    for i in range(n_steps + 1):
        ind = get_node_index(np.array([i, 0]), n_steps)
        g.constrain_velocity(ind, np.array([0.0]*3))

    g.K = g.K.tocsc()
    g.M = g.M.tocsc()
    tau = g.estimate_tau()
    g.set_Newmark_params(0.5, 0.5, tau)
    with open(grid_path, 'wb') as ofile:
        pickle.dump(g, ofile, protocol=pickle.HIGHEST_PROTOCOL)

with open(grid_path, 'rb') as ifile:
    g = pickle.load(ifile)


for i in range(n_steps):
    ind = get_elem_index([[i, n_steps - 1]], n_steps)
    for j in ind:
        g.elements[j].b = np.array([0.0, 0.0, 1.0])*f_mag

g.assemble_f()
g.dump_vtk_grid(dmp_path + 'f0')
for i in range(iters_strike):
    for _ in range(ipf):
        g.iteration_Newmark()
    g.dump_vtk_grid(dmp_path + f'f{i + 1}')

for i in range(n_steps):
    ind = get_elem_index([[i, n_steps - 1]], n_steps)
    for j in ind:
        g.elements[j].b = np.array([0.0, 0.0, 0.0])
g.assemble_f()

for i in range(iters_after):
    for _ in range(ipf):
        g.iteration_Newmark()
    g.dump_vtk_grid(dmp_path + f'f{i + 1 + iters_strike:}')
