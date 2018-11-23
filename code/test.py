import code.membrane as mb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


n_x, n_y = 20, 20
I = (n_x+1)*n_y + 2*n_x
g = mb.generate_uniform_grid(1.0, 1.0, n_x, n_y, 1e9, 0.4, 0.001, 900)
#g.a[3*I:3*(I + 1)] = [0.0, 0.0, 0.1]
g.elements[I].b = np.array([0.0, 0.0, 1e11])
g.ready()
g.dump_vtk_grid('../dmp/mov/usg0')
for i in range(1,100):
    g.iteration()
    g.dump_vtk_grid(f'../dmp/mov/usg{i:d}')

with open('../dmp/matr.txt', 'a') as output:
    for el in g.elements :
        output.write(np.array2string(el.DB, precision=1, separator=', '))
        output.write('\n')
# fig, axs = plt.subplots()
# polys = g.get_matplotlib_polygons()
# for p in polys:
#     axs.add_patch(p)
# axs.set_xlim((-.5, 15.5))
# axs.set_ylim((-.5, 15.5))
#
#
# fig.savefig('../dmp/patch.png', fmt='png')
