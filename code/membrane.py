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
        self.b = np.zeros([3])

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

        self.P = []                                         # some inverted matrix, may be will be removed if time
                                                            # integration method changes; now i want to test it so
                                                            # use the same thing i used in 1D test:
                                                            # P = (M + \tau**2 K)^(-1)
        self.tau = 0.0                                      # optimal timestep (that everything converges)

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
        point_coords = np.zeros([self.n_nodes, 3])
        point_coords += self.a.reshape([self.n_nodes, 3])
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
            if elem.S < 0.0:
                elem.node_ind = elem.node_ind[::-1]             # if for some reason they are in clockwise order instead
                elem.S *= -1.0                                  # of counter-clockwise, change order and fix area

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
                                [0, 0, 0]])
                B_2 = np.array([[gamma, beta, 0],
                                [0, 0, beta],
                                [0, 0, gamma]])
                DB_1 = D_1@B_1
                DB_2 = D_2@B_2
                elem.DB[:, 3*i:3*(i+1)] = np.row_stack((DB_1, DB_2))

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
                                [0, 0, 0]])
                B_2 = np.array([[gamma, beta, 0],
                                [0, 0, beta],
                                [0, 0, gamma]])                                         # assuming DB was already counted
                B[:, 3*i:3*(i+1)] = np.row_stack((B_1, B_2))
            K_e = B.T @ elem.DB
            K_e *= 0.5*elem.S*elem.h                                                # local stiffness obtained
            #with open('../dmp/matr.txt', 'a') as output:
            #    output.write(np.array2string(K_e, precision=1))                                                      # projecting to global matrix
            #    output.write('\n\n\n')
            for i in range(3):
                for j in range(3):
                    I , J = elem.node_ind[i], elem.node_ind[j]
                    self.K[3*I:3*(I+1), 3*J:3*(J+1)] += K_e[3*i:3*(i+1), 3*j:3*(j+1)]

    def assemble_f(self):
        self.f.fill(0.0)
        for elem in self.elements:
            for i in range(3):
                I = elem.node_ind[i]
                self.f[3*I:3*(I+1)] -= elem.S*elem.h*elem.b/6.0                     # see eq. 1.5.3

    def assemble_M(self):
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
            V = []
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

    def get_inv_matrix(self, tau : np.float64):                                     # getting matrix P = (M + \tau^2K)^-1
        self.P = np.linalg.inv(self.M + self.K*tau**2)

    def estimate_tau(self):
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
        self.tau = self.estimate_tau()
        self.get_inv_matrix(self.tau)

    def set_sigma(self):
        for elem in self.elements:
            I, J, K = elem.node_ind
            a_e = np.concatenate([self.a[3*I:3*(I+1)],
                                   self.a[3*J:3*(J+1)],
                                   self.a[3*K:3*(K+1)]])                            # getting displacement vector from global array
            elem.sigma = elem.DB @ a_e

    def iteration(self):
        self.assemble_f()
        self.v_a = self.P @ (self.M @ self.v_a - self.tau*(self.K @ self.a + self.f))
        self.a += self.v_a*self.tau
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
