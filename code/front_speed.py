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
strike_pos = np.array([size, size])/2


g = mb.generate_uniform_grid(size, size, n_0, n_0, E, nu, h, rho)  # created a grid with given params
g.ready()  # preparing grid
strike_ind = (get_node_index(strike_point,  n_0))

g.constrain_velocity(strike_ind, np.array([0.0, 0.0, 1.0]) * spd)
g.K = g.K.tocsc()
g.M = g.M.tocsc()
g.a_tt = sp.linalg.spsolve(g.M, -(g.K.dot(g.a) + g.f))  # count acceleration to satisfy equation

tau = g.estimate_tau()/4

g.set_Newmark_noinverse(0.5, 0.5, tau)

#n_plots = int(input('Number of plots: '))
n_iter = int(input('Number of iterations: '))
tp = np.dtype([('t', np.float64),
               ('r_peak', np.float64),
               ('r_inter', np.float64)])
results = np.array([], dtype=tp)

for j in range(n_iter):
    g.iteration_Newmark_noinverse()

    radial = g.get_radial_distribution(lambda u, v : [u[2], np.linalg.norm(v, ord=2)], center_cord=strike_pos)
    r_down = np.min(radial[0, radial[1] <= 0])
    r_up = np.min(radial[0, np.logical_and(radial[1] >= 0, radial[0] > r_down)])

    cone = radial[:, radial[0] <= (r_down + r_up)/2]
    p = np.polyfit(cone[0], cone[1], deg=1)
    line = np.poly1d(p)
    intersection = -p[1]/p[0]
    results = np.append(results, np.array([(tau*(j + 1.0), (r_down + r_up)/2, intersection)], dtype=tp))

fig, axs = plt.subplots(1, 2, sharex=True)
ax = axs[0]
ax.plot(radial[0], radial[1], 'bs')
ax.axvline(r_down, color='blue')
ax.axvline(r_up, color='blue')
ax.axvline((r_down + r_up)/2, color='green')
ax.axvline(intersection, color='red')
ax.grid(True)
ax = axs[1]
ax.plot(radial[0], radial[2], 'rs')
ax.grid(True)
plt.show()


p_peak = np.polyfit(results['t'], results['r_peak'], deg=1)
p_inter = np.polyfit(results['t'], results['r_inter'], deg=1)
line_peak = np.poly1d(p_peak)
line_inter = np.poly1d(p_inter)

fig, ax = plt.subplots(1, 1)
ax.plot(results['t'], results['r_peak'], 'b^')
ax.plot(results['t'], results['r_inter'], 'r^')
ax.plot(results['t'], line_peak(results['t']), 'b--')
ax.plot(results['t'], line_inter(results['t']), 'r--')
plt.show()






