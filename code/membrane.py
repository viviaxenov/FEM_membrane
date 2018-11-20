import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# tbc = to be counted


class Element:
    """Triangle finite element"""
    def __init__(self, node_ind : (int, int, int),
                        E : np.float64, nu : np.float64,
                        h : np.float64, rho : np.float64):
        self.node_ind = node_ind                            # tuple of 3 node indices
        self.E = E                                          # Young's modulus
        self.nu = nu                                        # Poisson's ratio
        self.h = h                                          # thickness
        self.rho = rho                                      # density

        self.DB = np.zeros([6, 9], np.float64)              # matrix connecting stress and nodal displacements; tbc
        self.S = 0.0                                        # doubled element area; tbc

    def to_string(self):
        return f"{self.node_ind}\nE = {self.E:.3f}, nu = {self.nu:.3f}\nh = {self.h:.3f}, rho = {self.rho:.3f}"


class Grid:
    def __init__(self, n_nodes : int):
        self.n_nodes = n_nodes                              # number of nodes
        self.a = np.zeros([3*n_nodes])                      # vector of nodal displacements
        self.x_0 = np.zeros([n_nodes])
        self.y_0 = np.zeros([n_nodes])                      # initial positions
        self.elements = []                                  # list of elements, will be filled later
        self.K = np.zeros([3*n_nodes, 3*n_nodes])           # global stiffness matrix
        self.M = np.zeros([3*n_nodes, 3*n_nodes])           # global mass matrix
        self.f = np.zeros([3*n_nodes])                      # load vector

    def add_elem(self, elem : Element):
        if(max(*elem.node_ind) >= self.n_nodes) or (min(*elem.node_ind) < 0):
            raise ValueError("Element's node indices out of range")
        if elem.E < 0.0 or elem.nu < 0 or elem.nu > 0.5 or elem.rho <= 0 or elem.h <= 0:
            raise ValueError("Wrong element params:\n" + elem.to_string())
        self.elements.append(elem)

    def get_matplotlib_polygons(self) -> [Polygon]:
        polys = []
        for elem in self.elements:
            vertices = []
            for i in elem.node_ind:
                vertices.append([self.x_0[i], self.y_0[i]])
            polys.append(Polygon(vertices, closed=True, fill=False, edgecolor='b'))
        return polys

    def set_S(self):
        for elem in self.elements:
            i, j, k = elem.node_ind
            delta = np.array([  [ 1.0, self.x_0[i] , self.y_0[i]],
                                [ 1.0, self.x_0[j] , self.y_0[j]],
                                [ 1.0, self.x_0[k] , self.y_0[k]] ])
            elem.S = np.linalg.det(delta)




