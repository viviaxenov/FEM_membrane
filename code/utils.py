import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

#import matplotlib.pyplot as plt

import membrane as mb

def get_node_index(node_cords : np.ndarray, n_x : int):
    return node_cords[0] + (n_x + 1)*node_cords[1]                    #


def split_elems(elem_cords):
    new_elem_cords = []
    for i in range(len(elem_cords)):
        x, y = elem_cords[i][0], elem_cords[i][1]
        new_elem_cords += [[2*x, 2*y],
                           [2*x + 1, 2*y],
                           [2*x, 2*y + 1],
                           [2*x + 1, 2*y + 1]]
    return new_elem_cords


def get_elem_index(elem_coords, n_x : int):
    elem_index = []
    for cord in elem_coords:
        x, y = cord[0], cord[1]
        elem_index += [2*(x + n_x*y), 2*(x + n_x*y) + 1]
    return elem_index


def measure_front_speed(g : mb.Grid, n_iter: int, strike_pos: np.ndarray, eps: np.float64=3) -> np.float64:
    g.a_tt = sp.linalg.spsolve(g.M, -(g.K.dot(g.a) + g.f))  # count acceleration to satisfy equation

    tp = np.dtype([('t', np.float64),
                   ('r_inter', np.float64)])
    results = np.array([], dtype=tp)

    for j in range(n_iter):
        g.iteration_Newmark_noinverse()

        radial = g.get_radial_distribution(lambda u, v : [u[2], np.linalg.norm(v, ord=2)], center_cord=strike_pos)
        w_thr = eps*np.linalg.norm(radial[1], ord=2)/radial[1].shape[0]
        r_min = np.min(radial[0, np.abs(radial[1]) < w_thr])

#        try:
#            r_min = np.min(radial[0, np.abs(radial[1]) < w_thr])
#        except Exception as err:
#            plt.plot(radial[0], radial[1], 'b^')
#            plt.axhline(w_thr)
#            plt.axhline(-w_thr)
#            plt.grid()
#            plt.show()
#            print(f'n_iter = {0:d}\n'.format(j))
#            raise err

        cone = radial[:, radial[0] <= r_min]
        p = np.polyfit(cone[0], cone[1], deg=1)

        intersection = -p[1]/p[0]
        results = np.append(results, np.array([(g.tau*(j + 1.0), intersection)], dtype=tp))

    p = np.polyfit(results['t'], results['r_inter'], deg=1)
    return p[0]



