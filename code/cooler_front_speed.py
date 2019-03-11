import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

import code.membrane as mb
import code.utils as util

n_0 = 20
division = 2
n_iter = 2*division*n_0


samples = 10

Es = np.linspace(1e6, 1e7, samples, endpoint=True)
speeds = np.linspace(10, 100, samples, endpoint=True)

front_speeds = np.zeros([Es.shape[0], speeds.shape[0]])

size = 1.0
nu = 0.45
h = 0.001
rho = 900
magnitude = 1e8
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
        try:
            front_speeds[i, j] = util.measure_front_speed(g, n_iter, strike_pos)
        except Exception as err:
            print(i, j)
            raise err
        g.discard_displacement()
    g.K *= factors[i]
    tau /= np.sqrt(factors[i])


np.savez('front_speed_data', front_speeds=front_speeds, E=Es, speeds=speeds)






