import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import code.membrane as mb

grid_path = '../pickles/random'
dump_path = '../res/random/'
if not os.path.exists(grid_path):
    mb.store_random_grid(grid_path, 20, 20, 4000, 1e6, 0.3, 1.0e-3, 900.0, disp=0.17)

g: mb.Grid
with open(grid_path, 'rb') as ifile:
    g = pickle.load(ifile)

if not os.path.exists(dump_path):
    os.mkdir(dump_path)

center_ind = np.argmin((g.x_0 - 0.5)**2 + (g.y_0 - 0.5)**2, axis=0)

g.constrain_velocity(center_ind, np.array([.0, .0, 1.])*20.0)
tau = g.estimate_tau()/4
g.set_Newmark_params(0.5, 0.5, tau)
g.dump_vtk_grid(f"{dump_path}f0")


for i in range(60):
    g.iteration_Newmark()
    g.dump_vtk_grid(f"{dump_path}f{i + 1:d}")

