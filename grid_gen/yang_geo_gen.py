import pygmsh
import numpy as np
import meshio

from scipy.spatial.transform import Rotation as R

def add_primary_hole(geo: pygmsh.opencascade.Geometry, l1, r1, a,  x_0=0., y_0=0., lcar=0.1, rotation_deg=45):
    """
        Returns gmsh geometry string for a hole that is always 
        inside lattice square (45deg to x axis in original paper). 
        Resulting coordinates are so that lattice element bottom left corner has cords (x_0, y_0)
        Args:
           l1 - length
           r1 - width and circle arc radius 
           a - lattice element characteristic size
           lcar - mesh density around hole
    """
    rect = np.array([
            [-r1, -l1/2, 0.0],
            [r1, -l1/2, 0.0],
            [r1, l1/2, 0.0],
            [-r1, l1/2, 0.0],
        ])

    r = R.from_euler('z', -rotation_deg, degrees=True)
    for el in [rect]:
        el[:] = r.apply(el)
        el += np.array([x_0, y_0, 0.0])
        el += np.array([a/2, a/2, 0.0])


    poly = geo.add_polygon(rect, lcar=lcar)

    c_top = geo.add_disk((rect[2] + rect[3])/2, r1)
    c_bot = geo.add_disk((rect[0] + rect[1])/2, r1)

    hole = geo.boolean_union([c_top, poly, c_bot])

    #centers
    return hole

def add_secondary_holes(geo: pygmsh.opencascade.Geometry, l2, r2, a,  x_0=0., y_0=0., lcar=0.1, rotation_deg=45):
    left_corner = np.array([-a/np.sqrt(2), 0.0, 0.0])
    right_corner = - left_corner

    base_rect = np.array([
                    [0.0, -r2, 0.0],
                    [l2, -r2, 0.0],
                    [l2, r2, 0.0],
                    [0.0, r2, 0.0],
                        ])
    left_rect = base_rect - np.array([a/np.sqrt(2), 0.0, 0.0])
    left_rect = np.roll(left_rect, 2, axis=0)

    right_rect = base_rect + np.array([a/np.sqrt(2) - l2, 0.0, 0.0])

    r = R.from_euler('z', -rotation_deg, degrees=True)
    for el in [left_rect, right_rect]:
        el[:] = r.apply(el)
        el += np.array([x_0, y_0, 0.0])
        el += np.array([a/2, a/2, 0.0])
    
    holes = []

    for rect in [left_rect, right_rect]:

        poly = geo.add_polygon(rect, lcar=lcar)

        circle = geo.add_disk((rect[3] + rect[0])/2, r1)

        holes.append(geo.boolean_union([poly, circle]))
    return holes

def add_lattice_element(geo: pygmsh.opencascade.Geometry, l1, r1, l2, r2, a,  x_0=0., y_0=0., lcar=0.1, rotation_deg=45):
    holes = [add_primary_hole(geo, l1, r1, a,  x_0, y_0, lcar, rotation_deg)] \
            + add_secondary_holes(geo, l2, r2, a, x_0, y_0, lcar, rotation_deg)
    return holes


d = 50.*1e-3
a = 3.5*1e-3
dx = 10*1e-3
dx_trans = 8*1e-3
Lx_add = 40*1e-3
dy = 50*1e-3
l1 = 0.73*a
l2 = 0.20*a
r1 = 0.1*a
r2 = 0.1*a

n_x = 9
n_y = 14

lcar_external = 1*1e-2
lcar_holes = 1e-3

geo = pygmsh.opencascade.Geometry()

holes = []

lattice = []

for i_x in range(n_x):
    for i_y in range(n_y):
        x_0 = i_x*a
        y_0 = i_y*a
        holes += add_lattice_element(geo, l1, r1, l2, r2, a,  x_0=x_0, y_0=y_0, lcar=lcar_holes)

#lattice_area = geo.add_polygon([
#                                [0.0, 0.0, 0.0],
#                                [n_x*a, 0.0, 0.0],
#                                [n_x*a, n_y*a, 0.0],
#                                [0.0, n_y*a, 0.0],
#                            ], lcar=lcar_holes)

#perf = geo.boolean_intersection([lattice_area] + holes)

poly = geo.add_polygon([
            [-dx, -dy, 0.0],
            [n_x*a + dx + Lx_add, -dy, 0.0],
            [n_x*a + dx + Lx_add, n_y*a + dy, 0.0],
            [-dx, n_y*a + dy, 0.0],
            ], 
            lcar=lcar_external, 
            )

transmitter_area = geo.add_polygon([
                                        [-dx_trans, -2*a, 0.0],
                                        [-dx_trans, (n_y+2)*a, 0.0],
                                        [-dx, (n_y+2)*a, 0.0],
                                        [-dx, -2*a, 0.0],
                                    ],
                                    lcar=lcar_holes,
                                    )

#base = geo.boolean_union([transmitter_area, poly])
#
#result = geo.boolean_difference([base], holes)
tmp = geo.boolean_difference([poly], holes)
res = geo.boolean_union([tmp, transmitter_area])

with open('wt.geo', 'w') as ofile:
    ofile.write(geo.get_code())

mesh = ""
try:
    mesh = pygmsh.generate_mesh(geo, verbose=True, dim=2)
except Exception as e:
    pass
finally:
    meshio.write("test.vtk", mesh)


