import membrane as mb

import numpy as np

import time

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


E = 1e6
nu = 0.45
h = 0.001
rho = 900
magnitude = 1e8
size = 1.0


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
loaded_elem_cords = [[(n_0 - 1)//2, (n_0 - 1)//2]]

sample_u = []
sample_v = []


tp = np.dtype([('k', np.int32), ('n_0', np.int32),
                        ('iterations', np.int32), ('t_ready', np.float64),
                        ('t_Newmark', np.float64), ('t_iters', np.float64)])


stats = np.array([(-1, -1, -1, -1, -1, -1)], dtype=tp)

for k in range(splits):
    g = mb.generate_uniform_grid(size, size, n_0, n_0, E, nu, h, rho)       # created a grid with given params
    elem_ind = get_elem_index(loaded_elem_cords, n_0)
    for i in elem_ind:
        g.elements[i].b += magnitude*np.array([0.0, 0.0, 1.0])              # added normal load to elements

    start = time.time()
    g.ready()                                                               # preparing grid
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

    g.dump_vtk_grid('../res/conv/convergence{0:d}'.format(k))

    sample_indices = get_node_index(sample_cords, n_0)
    s_u = []
    s_v = []
    for index in sample_indices:                                            # getting samples
        s_u = np.append(s_u, g.a[3*index:3*(index + 1)])
        s_v = np.append(s_v, g.a_t[3*index:3*(index + 1)])
#    sample_u.append(s_u)
#    sample_v.append(s_v)
    np.save('../res/conv/sample_u{0:d}'.format(k), s_u)
    np.save('../res/conv/sample_v{0:d}'.format(k), s_v)

    stats = np.append(stats, np.array([(k, n_0, iterations, t_ready, t_Newmark, t_iters)], dtype=tp))
    np.save('../res/conv/stats', stats)

    n_0 *= 2
    sample_cords *= 2
    loaded_elem_cords = split_elems(loaded_elem_cords)
    iterations *= 2
#sample_u = np.array(sample_u)
#sample_v = np.array(sample_v)
#
#np.save('../res/conv/sample_u', sample_u)
#np.save('../res/conv/sample_v', sample_v)

