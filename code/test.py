import code.membrane as mb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


g = mb.Grid(4)
g.x_0[0:4] = [0.0, 3.0, 0.0, 3.0]
g.y_0[0:4] = [0.0, 0.0, 4.0, 4.0]
e = mb.Element((0, 1, 2), 1e6, 0.3, 0.01, 4000)
g.add_elem(e)
e = mb.Element((1, 3, 2), 1e6, 0.3, 0.01, 4000)
g.add_elem(e)
g.set_S()
g.set_BDmatrix()
g.a[3:6] = [0.0, 1.0, 1.0]
g.v_a[3:6] = [1.0, 1.0, 1.0]
with open('../dmp/matr.txt', 'w') as output:
    for el in g.elements :
        output.write(np.array2string(el.DB, precision=1))
        output.write('\n')
g.dump_vtk_grid('../dmp/usg')
fig, axs = plt.subplots()
polys = g.get_matplotlib_polygons()
for p in polys:
    print(p)
    axs.add_patch(p)
axs.set_xlim((-6, 6))
axs.set_ylim((-6, 6))
fig.savefig('../dmp/patch.png', fmt='png')
