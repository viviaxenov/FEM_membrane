import code.membrane as mb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


g = mb.generate_uniform_grid(10.0, 15.0, 10, 15, 1e5, 0.3, 0.1, 4000)
g.set_S()
g.set_DBmatrix()
g.assemble_K()

node = 35
g.a[3*(node):3*(node + 1)] = [0.3, 0.3, 0.0]
g.set_sigma()
g.dump_vtk_grid('../dmp/usg')

# with open('../dmp/matr.txt', 'a') as output:
#     for el in g.elements :
#         output.write(np.array2string(el.DB, precision=1))
#         output.write('\n')
#     output.write(np.array2string(g.K, precision=1))
# fig, axs = plt.subplots()
# polys = g.get_matplotlib_polygons()
# for p in polys:
#     axs.add_patch(p)
# axs.set_xlim((-.5, 15.5))
# axs.set_ylim((-.5, 15.5))
#
#
# fig.savefig('../dmp/patch.png', fmt='png')
