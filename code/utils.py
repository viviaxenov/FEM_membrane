import numpy as np

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
