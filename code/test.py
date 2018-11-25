import code.membrane as mb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


n_x, n_y = 20, 20
I = (n_x+1)*n_y + 2*n_x
g = mb.generate_uniform_grid(1.0, 1.0, n_x, n_y, 1e9, 0.4, 0.001, 900)
#g.a[3*I:3*(I + 1)] = [0.0, 0.0, 0.1]
magnitude = 1e11
angle = np.pi/4
force = magnitude*np.array([0.0, np.cos(angle), np.sin(angle)])
g.elements[I].b = force
g.ready()
g.dump_vtk_grid('../dmp/mov/usg0')
for i in range(1, 150):
#   g.iteration_Newmark(g.tau, 0.5, 0.5)            # beta_2 >= beta_1 >= 0.5 makes Newmark stable regardless of tau
                                                    # if beta_2 = 0.5 it's at least 2nd order
    g.iteration()
    g.dump_vtk_grid(f'../dmp/mov/usg{i:d}')

#with open('../dmp/matr.txt', 'a') as output:
#    for el in g.elements :
#        output.write(np.array2string(el.DB, precision=1, separator=', '))
#        output.write('\n')
# fig, axs = plt.subplots()
# polys = g.get_matplotlib_polygons()
# for p in polys:
#     axs.add_patch(p)
# axs.set_xlim((-.5, 15.5))
# axs.set_ylim((-.5, 15.5))
#
#
# fig.savefig('../dmp/patch.png', fmt='png')
