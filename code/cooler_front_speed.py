import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

import membrane as mb
import utils as util

import os, shutil

size = 1.0
nu = 0.45
h = 0.001
rho = 900

samples = 20
n_0 = 50
division = 4
n_iter = division*n_0//2

if os.path.exists("../res/data"):
    shutil.rmtree("../res/data", ignore_errors=True)

path = ['../res/data/small/', '../res/data/big/']
params = [  (np.linspace(1, 10, samples, endpoint=True), np.linspace(1e6, 1e8, samples, endpoint=True)), # v/c ~ 1e-4 - 1e-2
            (np.linspace(10, 100, samples, endpoint=True), np.linspace(1e4, 1e6, samples, endpoint=True)) # v/c ~ 1e-2 - 1
            ]
for k in range(2):
    dirpath = path[k] + 'vtk/'
    os.makedirs(dirpath)

    speeds, Es = params[k]

    front_speeds = np.zeros([Es.shape[0], speeds.shape[0]])

    strike_ind = util.get_node_index(np.array([n_0, n_0], dtype=np.int32)//2, n_0)
    strike_pos = np.array([size, size])/2

    factors = np.roll(Es, -1)/Es

    g = mb.generate_uniform_grid(size, size, n_0, n_0, Es[0], nu, h, rho)
    g.ready()
    tau = g.estimate_tau()/division
    g.K = g.K.tocsc()
    g.M = g.M.tocsc()


    for i in range(Es.shape[0]):
        g.set_Newmark_noinverse(0.5, 0.5, tau)
        for j in range(speeds.shape[0]):
            g.constrain_velocity(strike_ind, np.array([0.0, 0.0, 1.0]) * speeds[j])
            front_speeds[i, j] = util.measure_front_speed(g, n_iter, strike_pos)
            #try:
            #    front_speeds[i, j] = util.measure_front_speed(g, n_iter, strike_pos)
            #except Exception as err:
            #    print(i, j)
            #    raise err
            g.dump_vtk_grid(dirpath + 'E={0:.1e}_v={1:.1e}'.format(Es[i], speeds[j]))
            g.discard_displacement()
        g.K *= factors[i]
        tau /= np.sqrt(factors[i])

    np.savez(path[k] + 'front_speed_data', front_speeds=front_speeds, E=Es, speeds=speeds)






