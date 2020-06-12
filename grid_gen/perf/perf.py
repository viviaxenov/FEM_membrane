import pygmsh
import numpy as np
import meshio
import sys



a = 8
b = 2
r = .5
d = 7
e = 6

l = 0.2

n_y = 3 
s = b/n_y
n_x = int(e/s)
r1 = 0.3*s


assert e < (d -r)

rho = 900.0

for name in ['perf', 'no_perf']:
    geo = pygmsh.built_in.Geometry()
    holes = []

    c_load = geo.add_circle([-d, 0., 0.], r, make_surface=True, lcar=2*l)
    c_constr = geo.add_circle([+d, 0., 0.], r, make_surface=False, lcar=l)

    S = a*b*4. - np.pi*r**2
    S_perf = S

    holes += [c_load, c_constr]

    if name == 'perf':
        for i in range(-n_x, n_x):
            for j in range(-n_y, n_y):
                c_perf = geo.add_circle([i*s + 0.5*s, j*s + 0.5*s, 0.], r1, make_surface=False, lcar=.5*l)
                holes.append(c_perf)
                S_perf -= np.pi*r1**2

    p = geo.add_polygon([
                            [-a, -b, 0],
                            [-a, b, 0],
                            [a, b, 0],
                            [a, -b, 0],
                                        ], 
                                        holes=holes, 
                                        lcar=l)

    print(rho*S_perf/S)
    geo.add_physical(c_load.plane_surface, label='load') 
    geo.add_physical(c_constr.line_loop.lines, label='hole') 

    geo.add_physical(p.surface, label='plate')
    with open(f'{name}.geo', 'w') as ofile:
        ofile.write(geo.get_code())

    for fmt in ['inp', 'vtk']:
        try:
            mesh = pygmsh.generate_mesh(geo, verbose=False, dim=2, mesh_file_type=fmt, msh_filename=f'{name}.{fmt}')
        except AssertionError:
            pass


