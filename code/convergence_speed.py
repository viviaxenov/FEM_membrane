import code.membrane as mb

import numpy as np

import time

from code.utils import *
import os
import shutil


E = 1e6
nu = 0.45
h = 0.001
rho = 900
magnitude = 1e8
size = 1.0
v_mag = 25.0


n_0 = 3                  # number of squares per side
                          # start with odd number to have middle element
iterations = 3           # number of iterations at initial step
splits = 6               # times the grid is split

n_0 = int(input('n_0 - initial squares per side: '))
iterations = int(input('Initial number of iterations: '))
splits = int(input('Number of grid splits: '))


x, y = [], []
for i in range(n_0 + 1):
    x += list(range(n_0 + 1))
    y += [i]*(n_0 + 1)
sample_cords = np.array([x, y])                                             # initial loaded elements and sample cords

sample_u = []
sample_v = []

theta = np.pi/6


tp = np.dtype([('k', np.int32), ('n_0', np.int32),
                        ('iterations', np.int32), ('t_ready', np.float64),
                        ('t_Newmark', np.float64), ('t_iters', np.float64)])


stats = np.array([(-1, -1, -1, -1, -1, -1)], dtype=tp)

dmp_path = "../res/conv/pi6_speed_finer/"
if not os.path.exists(dmp_path):
    os.makedirs(dmp_path)

for k in range(splits + 1):
    g = mb.generate_uniform_grid(size, size, n_0, n_0, E, nu, h, rho)       # created a grid with given params
    center_cord = np.array([n_0//2, n_0//2], dtype=int)
    center_ind = get_node_index(center_cord, n_0)

    start = time.time()
    g.ready()                                                               # preparing grid
    g.constrain_velocity(center_ind, np.array([0.0, np.sin(theta), np.cos(theta)])*v_mag)
    g.K = g.K.tocsc()
    g.M = g.M.tocsc()
    stop = time.time()
    t_ready = stop - start

    tau = g.estimate_tau()

    start = time.time()
    g.set_Newmark_noinverse(0.5, 0.5, tau)
    stop = time.time()
    t_Newmark = stop - start

    start = time.time()
    for i in range(iterations):                                             # iterate
        g.iteration_Newmark_noinverse()
    stop = time.time()
    t_iters = stop - start

    g.dump_vtk_grid(dmp_path + 'convergence{0:d}'.format(k))

    sample_indices = get_node_index(sample_cords, n_0)
    s_u = []
    s_v = []
    for index in sample_indices:                                            # getting samples
        s_u = np.append(s_u, g.a[3*index:3*(index + 1)])
        s_v = np.append(s_v, g.a_t[3*index:3*(index + 1)])
#    sample_u.append(s_u)
#    sample_v.append(s_v)
    np.save(dmp_path + 'sample_u{0:d}'.format(k), s_u)
    np.save(dmp_path + 'sample_v{0:d}'.format(k), s_v)

    stats = np.append(stats, np.array([(k, n_0, iterations, t_ready, t_Newmark, t_iters)], dtype=tp))
    np.save(dmp_path + 'stats', stats)

    n_0 *= 2
    iterations *= 2
#sample_u = np.array(sample_u)
#sample_v = np.array(sample_v)
#
#np.save('../res/conv/sample_u', sample_u)
#np.save('../res/conv/sample_v', sample_v)

