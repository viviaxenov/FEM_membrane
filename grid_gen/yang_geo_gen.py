import pygmsh
import numpy as np
import meshio

from scipy.spatial.transform import Rotation as R

def add_primary_hole(geo: pygmsh.built_in.Geometry, l1, r1, a,  x_0=0., y_0=0., lcar=0.1, rotation_deg=45):
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
    rect_top_arc = np.array([0.0, l1/2 + r1, 0.0])
    rect_bot_arc  = -rect_top_arc

    r = R.from_euler('z', -rotation_deg, degrees=True)
    for el in [rect, rect_top_arc, rect_bot_arc]:
        el[:] = r.apply(el)
        el += np.array([x_0, y_0, 0.0])
        el += np.array([a/2, a/2, 0.0])

    p1 = geo.add_point(rect[0], lcar=lcar)
    p2 = geo.add_point(rect[1], lcar=lcar)
    p3 = geo.add_point(rect[2], lcar=lcar)
    p4 = geo.add_point(rect[3], lcar=lcar)

    c_top = geo.add_point((rect[2] + rect[3])/2)
    c_bot = geo.add_point((rect[0] + rect[1])/2)

    rect_top_arc = geo.add_point(rect_top_arc, lcar=lcar)
    rect_bot_arc = geo.add_point(rect_bot_arc, lcar=lcar)

    hole = geo.add_line_loop([
                geo.add_line(p2,p3),
                geo.add_circle_arc(p3, c_top, rect_top_arc),
                geo.add_circle_arc(rect_top_arc, c_top, p4),
                geo.add_line(p4, p1),
                geo.add_circle_arc(p1, c_bot, rect_bot_arc),
                geo.add_circle_arc(rect_bot_arc, c_bot, p2),
            ])
    #centers
    return hole

def add_secondary_holes(geo: pygmsh.built_in.Geometry, l2, r2, a,  x_0=0., y_0=0., lcar=0.1, rotation_deg=45):
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

    right_arc = np.array([a/np.sqrt(2) - l2 - r2, 0.0, 0.0])
    left_arc = -right_arc


    r = R.from_euler('z', -rotation_deg, degrees=True)
    for el in [left_rect, right_rect, right_arc, left_arc]:
        el[:] = r.apply(el)
        el += np.array([x_0, y_0, 0.0])
        el += np.array([a/2, a/2, 0.0])
    
    holes = []

    for rect, arc in [(left_rect, left_arc), (right_rect, right_arc)]:
        p1 = geo.add_point(rect[0], lcar=lcar)
        p2 = geo.add_point(rect[1], lcar=lcar)
        p3 = geo.add_point(rect[2], lcar=lcar)
        p4 = geo.add_point(rect[3], lcar=lcar)

        arc_center = geo.add_point((rect[3] + rect[0])/2)

        rect_top_arc = geo.add_point(arc, lcar=lcar)

        hole = geo.add_line_loop([
                    geo.add_line(p1, p2),
                    geo.add_line(p2, p3),
                    geo.add_line(p3, p4),
                    geo.add_circle_arc(p4, arc_center, rect_top_arc),
                    geo.add_circle_arc(rect_top_arc, arc_center, p1),
                ])
        holes.append(hole)
    return holes

d = 20.
a = 3.5
dx = 10
Lx_add = 50
dy = 75
l1 = 0.73*a
l2 = 0.20*a
r1 = 0.1*a
r2 = 0.1*a

n_x = 9
n_y = 20

lcar_external = 18
lcar_holes = 0.9

geo = pygmsh.built_in.Geometry()

holes = []

for i_x in range(n_x):
    for i_y in range(n_y):
        x_0 = i_x*a
        y_0 = i_y*a
        hole = add_primary_hole(geo, l1, r1, a, x_0, y_0, lcar=lcar_holes)
        holes.append(hole)
for i_x in range(n_x):
    for i_y in range(n_y):
        x_0 = i_x*a + a/2
        y_0 = i_y*a + a/2
        hole = add_primary_hole(geo, l2, r2, a, x_0, y_0, lcar=lcar_holes, rotation_deg=-45)
        holes.append(hole)
        #holes += add_secondary_holes(geo, l2, r2, a, x_0, y_0, lcar=lcar_holes)


poly = geo.add_polygon([
            [-dx, -dy, 0.0],
            [n_x*a + dx + Lx_add, -dy, 0.0],
            [n_x*a + dx + Lx_add, n_y*a + dy, 0.0],
            [-dx, n_y*a + dy, 0.0],
            ], 
            holes=holes,
            lcar=lcar_external
            )


with open('rectangle.geo', 'w') as ofile:
    ofile.write(geo.get_code())

mesh = ""
try:
    mesh = pygmsh.generate_mesh(geo, verbose=True, dim=2)
except Exception as e:
    pass
finally:
    meshio.write("test.vtk", mesh)


