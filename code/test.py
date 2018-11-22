import code.membrane as mb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# g = mb.generate_uniform_grid(10.0, 15.0, 10, 15, 1e5, 0.3, 0.1, 4000)
n_x, n_y = 2, 3
I = n_x*n_y//2 + n_x//2
g = mb.generate_uniform_grid(np.float64(n_x), np.float64(n_y), n_x, n_y, 1e5, 0.3, 0.1, 4000)
# g.v_a[3*I:3*(I + 1)] = [0.0, 0.0, 0.1]
g.ready()
g.dump_vtk_grid('../dmp/usg0')
# for i in range(45):
#     g.iteration()
#     g.dump_vtk_grid(f'../dmp/mov/usg{i:d}')

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
