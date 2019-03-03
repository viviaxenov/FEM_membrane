import numpy as np
from matplotlib.patches import Polygon

from scipy import sparse as sp
import scipy.sparse.linalg

import numba as nb

import pyvtk as pvtk
# tbc = to be counted/coded/computed/


# tup_type = nb.typeof((1, 1, 1))
# matr_type = nb.typeof(np.zeros([6, 9], dtype=np.float64)) 
# vec_type = nb.typeof(np.zeros(3, dtype=np.float64))
# spec = [
#             ('node_ind', tup_type),
#             ('E', nb.float64),
#             ('nu', nb.float64),
#             ('h', nb.float64), 
#             ('rho', nb.float64),
#             ('b', vec_type),
#             ('DB', matr_type),
#             ('S', nb.float64),
#             ('sigma', nb.float64[:]),
#     ]
# 
# @nb.jitclass(spec)
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
        self.b = np.zeros(3, dtype=np.float64)

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

    #    def to_string(self):
    #        return f"{self.node_ind}\nE = {self.E:.3f}, nu = {self.nu:.3f}\nh = {self.h:.3f}, rho = {self.rho:.3f}"


class Grid:
    """2D membrane in 3D space model"""
    def __init__(self, n_nodes : int):
        self.n_nodes = n_nodes                              # number of nodes
        self.a = np.zeros([3*n_nodes])                      # vector of nodal displacements
        self.a_t = np.zeros([3 * n_nodes])                  # vector of nodal velocities
        self.a_tt = np.zeros([3 * n_nodes])                 # vector of nodal accelerations
        self.x_0 = np.zeros([n_nodes])
        self.y_0 = np.zeros([n_nodes])                      # initial positions
        self.elements = []                                  # list of elements, will be filled later
        self.K = sp.dok_matrix(
            (3*n_nodes, 3*n_nodes), dtype=np.float64)       # global stiffness matrix
        self.M = sp.dok_matrix(
            (3*n_nodes, 3*n_nodes), dtype=np.float64)       # global mass matrix
        self.f = np.zeros([3*n_nodes])                      # load vector

        self.P = []                                         # some inverted matrix, may be will be removed if time
                                                            # integration method changes; now i want to test it so
                                                            # use the same thing i used in 1D test:
                                                            # P = (M + \tau**2 K)^(-1)
        self.tau = 0.0                                      # optimal timestep (that everything converges)
        self.beta_1 = 0.0                                   # Newmark algorithm params
        self.beta_2 = 0.0
        self.A = []

    def add_elem(self, elem : Element):
        if(max(*elem.node_ind) >= self.n_nodes) or (min(*elem.node_ind) < 0):
            raise ValueError("Element's node indices out of range")
        if elem.E < 0.0 or elem.nu < 0 or elem.nu > 0.5 or elem.rho <= 0 or elem.h <= 0:
            raise ValueError("Wrong element params:\n" + elem.to_string())
        self.elements.append(elem)

    def get_elem_midpoint(self, elem_ind : int) -> np.ndarray:
        """Returns middle point of element"""
        indices = np.array(self.elements[elem_ind].node_ind)
        vertex_coords = np.vstak((self.x_0[indices], self.y_0[indices]))
        return vertex_coords.mean(axis=1)
        

    def get_matplotlib_polygons(self) -> [Polygon]:
        polys = []
        for elem in self.elements:
            vertices = []
            for i in elem.node_ind:
                vertices.append([self.x_0[i], self.y_0[i]])
            polys.append(Polygon(vertices, closed=True, fill=False, edgecolor='b'))
        return polys

    def dump_vtk_grid(self, path : str):
        point_coords = np.zeros([self.n_nodes, 3])
        point_coords += self.a.reshape([self.n_nodes, 3])
        point_coords[:, 0] += self.x_0
        point_coords[:, 1] += self.y_0
        point_velocities = self.a_t.reshape([self.n_nodes, 3])
        pd = pvtk.PointData(pvtk.Vectors(point_velocities, name='Velocity'))
        pd.append(pvtk.Scalars(point_coords[:, 2], name='z'))

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
        """"Calculates doubled area of each element; this will be used in further computations"""
        for elem in self.elements:
            i, j, k = elem.node_ind
            delta = np.array([  [ 1.0, self.x_0[i] , self.y_0[i]],
                                [ 1.0, self.x_0[j] , self.y_0[j]],
                                [ 1.0, self.x_0[k] , self.y_0[k]] ])
            elem.S = np.linalg.det(delta)
            if elem.S < 0.0:
                elem.node_ind = elem.node_ind[::-1]             # if for some reason they are in clockwise order instead
                elem.S *= -1.0                                  # of counter-clockwise, change order and fix area

    def set_DBmatrix(self):
        """"Calculates matrix linking stress and displacement for each element"""
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
                                [0, 0, 0]])
                B_2 = np.array([[gamma, beta, 0],
                                [0, 0, beta],
                                [0, 0, gamma]])
                DB_1 = D_1@B_1
                DB_2 = D_2@B_2
                elem.DB[:, 3*i:3*(i+1)] = np.row_stack((DB_1, DB_2))

    def assemble_K(self):                                                        # K = 0.5Sh*B.T @(DB) see eq 1.4.1
        """"Calculates global stiffness matrix for entire grid; used in both static and dynamic problems"""
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
                                [0, 0, 0]])
                B_2 = np.array([[gamma, beta, 0],
                                [0, 0, beta],
                                [0, 0, gamma]])                                         # assuming DB was already counted
                B[:, 3*i:3*(i+1)] = np.row_stack((B_1, B_2))
            K_e = B.T @ elem.DB
            K_e *= 0.5*elem.S*elem.h                                                # local stiffness obtained
            for i in range(3):
                for j in range(3):
                    I , J = elem.node_ind[i], elem.node_ind[j]
                    self.K[3*I:3*(I+1), 3*J:3*(J+1)] += K_e[3*i:3*(i+1), 3*j:3*(j+1)]

    def assemble_f(self):
        """"Calculates nodal force vector from elements' distributed force vectors"""
        self.f.fill(0.0)
        for elem in self.elements:
            for i in range(3):
                I = elem.node_ind[i]
                self.f[3*I:3*(I+1)] -= elem.S*elem.h*elem.b/6.0                     # see eq. 1.5.3

    def assemble_M(self):
        """"Assembles mass matrix used in dynamic problem"""
        for elem in self.elements:
            I, J, K = elem.node_ind

            x = self.x_0[[I, J, K]]
            y = self.y_0[[I, J, K]]
            x -= np.average(x)
            y -= np.average(y)                                                      # switching to barycenter cords

            A = x.T@x
            B = y.T@y
            C = x.T@y                                                               # coefs from mass matrix eq

            beta = -(np.roll(y,1) - np.roll(y,2))/elem.S                            # see eq 1.1.6
            gamma = (np.roll(x,1) - np.roll(x,2))/elem.S                            # x_k = np.roll(x,1)
            alpha = (np.roll(x,2)*np.roll(y,1) - np.roll(x,1)*np.roll(y,2))/elem.S  # x_j = np.roll(x,2)
#           this was used when we used a custom 'Kirchoff-Love element' which didn't seem to work
#           but maybe it'll be back
#            V = []
#            for i in range(3):
#                V_i = np.array([[0.0, 0.0, -beta[i]],
#                                [0.0, 0.0, -gamma[i]],
#                                [beta[i], gamma[i], 0.0]])
#                V.append(V_i)

            for i in range(3):
                for j in range(3):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    coef = 12.0*alpha[i]*alpha[j]                                   # see eq 1.6.3
                    coef += A*beta[i]*beta[j]
                    coef += B*gamma[i]*gamma[j]
                    coef += C*(beta[i]*gamma[j] + gamma[i]*beta[j])
                    coef /= 2.0
                    M_ij = np.eye(3)*coef #+ elem.h**2*(V[i].T@V[j])
                    M_ij *= elem.rho*elem.S*elem.h/12.0                             # local submatrix assembled,

                    self.M[3*I:3*(I+1), 3*J:3*(J+1)] += M_ij                        # mapping to global




    def constrain_velocity(self, node_index : int, velocity : np.ndarray):
        """Applies constraint a_t = velocity = const in poin node_index"""
        # TODO: verification 
        self.a_t[3*node_index : 3*(node_index + 1)] = velocity
        self.K[3*node_index : 3*(node_index + 1), : ] = 0
        self.M[3*node_index : 3*(node_index + 1), : ] = 0
        self.M[3*node_index : 3*(node_index + 1), 3*node_index : 3*(node_index + 1)] = np.eye(3)
        self.f[3*node_index : 3*(node_index + 1)] = np.zeros(3)
	
	
#    def constrain_cord(self, node_index : int):
#	"""Applies constraint a = 0 in point node_index"""	


    def get_inv_matrix(self, tau : np.float64):                                     # getting matrix P = (M + \tau^2K)^-1
        self.P = sp.linalg.inv(self.M + self.K*tau**2)

    def estimate_tau(self):
        """"Estimates time step that tau < diam/c, c ~ sqrt(E/rho) (that everything converges"""
        def tau(e : Element):
            return  np.sqrt(e.S*e.rho/e.E)                                          # this may be wrong in counting both diameter
        tau_max = 0.5*max([tau(e) for e in self.elements])                          # and c, but i want to test the thing already
        return tau_max

    def ready(self):
        self.set_S()
        self.set_DBmatrix()
        self.assemble_K()
        self.assemble_f()
        self.assemble_M()

        self.set_sigma()

    def set_sigma(self):
        """Calculates stress tensor from displacements for each element; needs DB to be already calculated"""
        for elem in self.elements:
            I, J, K = elem.node_ind
            a_e = np.concatenate([self.a[3*I:3*(I+1)],
                                   self.a[3*J:3*(J+1)],
                                   self.a[3*K:3*(K+1)]])                            # getting displacement vector from global array
            elem.sigma = elem.DB @ a_e

    def iteration(self):
        self.assemble_f()
        self.a_t = self.P.dot((self.M.dot(self.a_t) - self.tau * (self.K.dot(self.a) + self.f)))
        self.a += self.a_t * self.tau
        self.set_sigma()

    def set_Newmark_params(self, beta_1 : np.float64, beta_2 : np.float64, tau : np.float64):
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.A = sp.linalg.inv(self.M + 0.5*beta_2*tau**2*self.K)

    def iteration_Newmark(self):
        self.assemble_f()

        a_est = self.a + self.tau * self.a_t + 0.5 * (1 - self.beta_2) * self.tau ** 2 * self.a_tt
        v_est = self.a_t + (1 - self.beta_1) * self.tau * self.a_tt

        a_tt_next =  self.A.dot((-self.K.dot(a_est) - self.f))                   # actually we need f_n+1 here

        self.a_tt = a_tt_next
        self.a_t = v_est + self.beta_1 * self.tau * a_tt_next
        self.a = a_est + 0.5*self.beta_2*self.tau**2*a_tt_next

        self.set_sigma()

    def set_Newmark_noinverse(self, beta_1 : np.float64, beta_2 : np.float64, tau : np.float64):
        """Sets params for Newmark iteration without preliminary inverting A matrix"""
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.A = self.M + 0.5*beta_2*tau**2*self.K

    def iteration_Newmark_noinverse(self):
        """"Newmark iteration not using preliminary inverted matrix A (solving at each step)"""
        self.assemble_f()

        a_est = self.a + self.tau * self.a_t + 0.5 * (1 - self.beta_2) * self.tau ** 2 * self.a_tt
        v_est = self.a_t + (1 - self.beta_1) * self.tau * self.a_tt

        a_tt_next = sp.linalg.spsolve(self.A, (-self.K.dot(a_est) - self.f))                   # actually we need f_n+1 here

        self.a_tt = a_tt_next
        self.a_t = v_est + self.beta_1 * self.tau * a_tt_next
        self.a = a_est + 0.5*self.beta_2*self.tau**2*a_tt_next

        self.set_sigma()


def generate_uniform_grid(X : np.float64, Y : np.float64, n_x : int, n_y : int,
                        E : np.float64, nu : np.float64,
                        h : np.float64, rho : np.float64) -> Grid:
    res = Grid((n_x + 1)*(n_y + 1))

    def index(i, j):
        return (n_x + 1)*i + j

    dx = X/n_x
    dy = Y/n_y

    for i in range(n_y + 1):
        for j in range(n_x + 1):
            k = index(i, j)
            res.x_0[k], res.y_0[k] = dx*j, dy*i
    for i in range(n_y):
        for j in range(n_x):
            res.elements.append(Element((index(i, j), index(i, j + 1), index(i + 1, j + 1)), E, nu, h, rho))
            res.elements.append(Element((index(i, j), index(i + 1, j + 1), index(i + 1, j)), E, nu, h, rho))
    return res


#    /|
#   / +============================|
#  X       To be continued   | \ /||
#   \ +============================|
#    \|
