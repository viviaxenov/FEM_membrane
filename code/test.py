import code.membrane as mb
import numpy as np
import os

E = 1e6
nu = 0.45
h = 0.001
rho = 900
size_x = 1.0
size_y = 1.0
magnitude = 2e8


def test_sequence(dim: int, iter_per_frame: int, iters: int = 150, beta: np.float64 = None):
    n_x = dim
    n_y = dim
    normal = mb.generate_uniform_grid(size_x, size_y, n_x, n_y, E, nu, h, rho)
    skew = mb.generate_uniform_grid(size_x, size_y, n_x, n_y, E, nu, h, rho)
    free = mb.generate_uniform_grid(size_x, size_y, n_x, n_y, E, nu, h, rho)

    i, j = n_x//2, n_y//2
    Ind_vertex = (n_x + 1)*i + j
    Ind_elem = 2*(n_x*i + j)

    normal.elements[Ind_elem].b += magnitude*np.array([0.0, 0.0, 1.0])
    skew.elements[Ind_elem].b += magnitude*np.array([0.0, np.cos(np.pi/4), np.sin(np.pi/4)])
    free.a[3*Ind_vertex:3*(Ind_vertex + 1)] += np.array([0.0, 0.0, 0.05])

    tau = 0.0
    for g in [normal, skew, free]:
        g.ready()
        tau = g.estimate_tau()/iter_per_frame


    if beta is None:
        path = f'../res/simple_d{dim:d}_ipf{iter_per_frame:d}'
    else:
        path = f'../res/beta{beta:.1f}_d{dim:d}_ipf{iter_per_frame:d}'

    if not os.path.exists(path):
        os.mkdir(path)

    with open(f'{path:s}/params.tex', 'w') as tex_output:
        tex_output.write(   "\\begin{tabular}[h]{|l|c|}\n"
                            "\\hline\n"
                            "\\multicolumn{2}{|c|}{Material properties} \\\\ \n"
                            "\\hline\n"
                            f"$E,MPa$ & ${E/1e6 : .1f}$ \\\\ \n"
                            f"$\\nu$ & ${nu : .2f}$ \\\\ \n"
                            f"$\\rho, \\frac{{g}}{{cm^3}}$ & ${rho/1000 : .1f}$ \\\\ \n"
                            f"$h, mm$ & ${h * 1000 : .1f}$ \\\\ \n"
                            "\\hline\n"
                            "\\multicolumn{2}{|c|}{Model parameters}\\\\ \n"
                            "\\hline\n"
                            f"$\\tau, s$ & ${tau : .1e}$ \\\\ \n"
                            f"Nodes & ${n_x + 1}\\times{n_y + 1}$ \\\\ \n"
                            f"Size, $m$ & ${size_x : .1f} \\times {size_y : .1f}$"
                            f"Pressure, Pa & ${magnitude * h : .1e}$\\"
                            "\\hline\n"
                            "\\end{{tabular}}")

    names = {normal: 'normal', skew: 'skew', free: 'free'}
    for s in ['normal', 'skew', 'free']:
        dirpath = f'{path}/{s}'
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    for g in [normal, skew, free]:
        if beta is None:
            g.get_inv_matrix(tau)
            g.tau = tau
            g.dump_vtk_grid(f'{path}/{names[g]}/{names[g][0]}0')
            for k in range(1, iters + 1):
                for p in range(iter_per_frame):
                    g.iteration()
                g.dump_vtk_grid(f'{path}/{names[g]}/{names[g][0]}{k:d}')
        else:
            g.set_Newmark_params(0.5, beta, tau)
            g.dump_vtk_grid(f'{path}/{names[g]}/{names[g][0]}0')
            for k in range(1, iters + 1):
                for p in range(iter_per_frame):
                    g.iteration_Newmark()
                g.dump_vtk_grid(f'{path}/{names[g]}/{names[g][0]}{k:d}')

dim = 30
iters = 150
#test_sequence(dim, 1, iters)
#test_sequence(dim, 1, iters, beta=0.5)
#test_sequence(dim, 1, iters, beta=0.7)
#test_sequence(dim, 1, iters, beta=1.0)
#
#test_sequence(dim, 5, iters)
test_sequence(dim, 5, iters, beta=0.5)
#test_sequence(dim, 5, iters, beta=1.0)

#n_x, n_y = 20, 20
#g = mb.generate_uniform_grid(1.0, 1.0, n_x, n_y, 1e9, 0.4, 0.001, 900)
#i, j = 10, 10
##g.a[3*Ind_vertex:3*(Ind_vertex + 1)] = [0.0, 0.0, 0.1]
#magnitude = 1e11
#angle = np.pi/4
#force = magnitude*np.array([0.0, np.cos(angle), np.sin(angle)])
#g.elements[Ind_elem].b = force
#g.ready()
#
#beta_2 = 1.0
#scale = 1.0
#
#g.set_Newmark_params(0.5, beta_2, g.estimate_tau()/scale)
#path = '../dmp/Newmark'
#g.dump_vtk_grid(f'{path}/usg0')
#for i in range(1, 100):
#   g.iteration_Newmark()                            # beta_2 >= beta_1 >= 0.5 makes Newmark stable regardless of tau
#                                                    # if beta_1 == 0.5 it's at least 2nd order
#   g.dump_vtk_grid(f'{path}/usg{i:d}')


