import numpy as np
from matplotlib.patches import Polygon
import pyvtk as pvtk
# tbc = to be counted/coded/computed/


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
        self.sigma = np.zeros([6])                          # stress tensor

    def get_Dmatrices(self) -> (np.ndarray, np.ndarray):    # see eq. 1.3.2 from desc.
        """Returns elastic moduli matrix for an element"""
        D_2 = np.identity(3)*(1 - 2.0*self.nu)
        D_1 = np.full([3,3], self.nu) + D_2
        D_2 /= 2.0
        D = self.E/(1 + self.nu)/(1 - 2.0*self.nu)
        D_1 *= D
        D_2 *= D
        return D_1, D_2

    def to_string(self):
        return f"{self.node_ind}\nE = {self.E:.3f}, nu = {self.nu:.3f}\nh = {self.h:.3f}, rho = {self.rho:.3f}"


class Grid:
    def __init__(self, n_nodes : int):
        self.n_nodes = n_nodes                              # number of nodes
        self.a = np.zeros([3*n_nodes])                      # vector of nodal displacements
        self.v_a = np.zeros([3*n_nodes])                    # vector of nodal velocities
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

    def dump_vtk_grid(self, path : str):
        point_coords = self.a.reshape([self.n_nodes, 3])
        point_coords[:, 0] += self.x_0
        point_coords[:, 1] += self.y_0
        point_velocities = self.v_a.reshape([self.n_nodes, 3])
        pd: pvtk.PointData = pvtk.PointData(pvtk.Vectors(point_velocities, name='Velocity'))

        triangles = []
        sigmas = []

        for elem in self.elements:
            triangles.append(list(elem.node_ind))
            sigmas.append(elem.sigma)
        cd = []
        names = ['sigma_xx', 'sigma_yy', 'sigma_zz', 'tau_xy', 'tau_yz', 'tau_xz']
        sigmas = np.array(sigmas)
        for i in range(6):
            cd.append(pvtk.Scalars(sigmas[:, i], name=names[i]))
        usg = pvtk.UnstructuredGrid(point_coords, triangle=triangles)
        vtk = pvtk.VtkData(usg, pd)
        for e in cd:
            vtk.cell_data.append(e)
        vtk.tofile(path, 'binary')

    def set_S(self):
        for elem in self.elements:
            i, j, k = elem.node_ind
            delta = np.array([  [ 1.0, self.x_0[i] , self.y_0[i]],
                                [ 1.0, self.x_0[j] , self.y_0[j]],
                                [ 1.0, self.x_0[k] , self.y_0[k]] ])
            elem.S = np.linalg.det(delta)


    def set_DBmatrix(self):
        for elem in self.elements:
            D_1, D_2 = elem.get_Dmatrices()
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                I, J, K = elem.node_ind[i], elem.node_ind[j], elem.node_ind[k]  # indices in external massive
                # TODO: omg this is so ugly and non-Python, need to find better solution
                beta = -(self.y_0[K] - self.y_0[J])/elem.S
                gamma = (self.x_0[K] - self.x_0[J])/elem.S
                B_1 = np.array([[beta, 0, 0],                                   # see eq 1.2.4
                                [0, gamma, 0],
                                [beta, gamma, 0]])
                B_2 = np.array([[gamma, beta, 0],
                                [0, 0, 0],
                                [0, 0, 0]])
                DB_1 = D_1@B_1
                DB_2 = D_2@B_2
                elem.DB[0:3, 3*i:3*(i+1)] = DB_1
                elem.DB[3:6, 3*i:3*(i+1)] = DB_2

    def assemble_K(self):                                                        # K = 0.5Sh*B.T @(DB) see eq 1.4.1
        for elem in self.elements:
            B = np.zeros([6, 9])
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                I, J, K = elem.node_ind[i], elem.node_ind[j], elem.node_ind[k]  # indices in external massive
                # TODO: omg this is so ugly and non-Python, need to find better solution
                beta = -(self.y_0[K] - self.y_0[J])/elem.S
                gamma = (self.x_0[K] - self.x_0[J])/elem.S
                B_1 = np.array([[beta, 0, 0],                                       # see eq 1.2.4
                                [0, gamma, 0],
                                [beta, gamma, 0]])
                B_2 = np.array([[gamma, beta, 0],
                                [0, 0, 0],
                                [0, 0, 0]])                                         # assuming DB was already counted
                B[:, 3*i:3*(i+1)] = np.row_stack((B_1, B_2))
            K_e = B.T @ elem.DB
            K_e *= 0.5*elem.S*elem.h                                                # local stiffness obtained
                                                                                    # projecting to global matrix
            for i in range(3):
                for j in range(3):
                    I , J = elem.node_ind[i], elem.node_ind[j]
                    self.K[3*I:3*(I+1), 3*J:3*(J+1)] += K_e[3*i:3*(i+1), 3*j:3*(j+1)]


    def set_sigma(self):
        for elem in self.elements:
            I, J, K = elem.node_ind
            a_e = np.concatenate([self.a[3*I:3*(I+1)],
                                   self.a[3*J:3*(J+1)],
                                   self.a[3*K:3*(K+1)]])                            # getting displacement vector from global array
            elem.sigma = elem.DB @ a_e



#    /|
#   / +============================|
#  X       To be continued   | \ /||
#   \ +============================|
#    \|
