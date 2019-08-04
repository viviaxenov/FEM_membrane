from typing import List

import numpy as np
from matplotlib.patches import Polygon

from scipy.spatial import Delaunay
from scipy import sparse as sp
import scipy.sparse.linalg

import sympy

import warnings
import pickle

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

    def __init__(self, node_ind: (int, int, int),
                 D: np.ndarray,
                 h: np.float64, rho: np.float64):
        self.node_ind = node_ind  # tuple of 3 node indices
        self.h = h  # thickness
        self.rho = rho  # density
        self.b = lambda t: np.zeros(3, dtype=np.float64)

        self.D = D  # tensor of elastic constants (6x6)

        self.DB = np.zeros([6, 9], np.float64)  # matrix connecting stress and nodal displacements; tbc
        self.S = 0.0  # doubled element area; tbc
        self.sigma = np.zeros([6])  # stress tensor

    def get_Dmatrices(self) -> (np.ndarray, np.ndarray):  # see eq. 1.3.2 from desc.
        """Returns elastic moduli matrix for an element"""
        D_2 = np.identity(3) * (1 - 2.0 * self.nu)
        D_1 = np.full([3, 3], self.nu) + D_2
        D_2 /= 2.0
        D = self.E / (1 + self.nu) / (1 - 2.0 * self.nu)
        D_1 *= D
        D_2 *= D
        return D_1, D_2

    def to_string(self):
        return f"{self.node_ind}\nE = {self.E:.3f}, nu = {self.nu:.3f}\nh = {self.h:.3f}, rho = {self.rho:.3f}"


class Grid:
    """2D membrane in 3D space model"""

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes  # number of nodes
        self.a = np.zeros([3 * n_nodes])  # vector of nodal displacements
        self.a_t = np.zeros([3 * n_nodes])  # vector of nodal velocities
        self.a_tt = np.zeros([3 * n_nodes])  # vector of nodal accelerations
        self.x_0 = np.zeros([n_nodes])
        self.y_0 = np.zeros([n_nodes])  # initial positions
        self.elements = []  # list of elements, will be filled later
        self.K = sp.dok_matrix(
            (3 * n_nodes, 3 * n_nodes), dtype=np.float64)  # global stiffness matrix
        self.M = sp.dok_matrix(
            (3 * n_nodes, 3 * n_nodes), dtype=np.float64)  # global mass matrix
        self.f = np.zeros([3 * n_nodes])  # load vector

        self.t = 0.
        self.tau = None  # optimal timestep (that everything converges)
        self.beta_1 = 0.0  # Newmark algorithm params
        self.beta_2 = 0.0
        self.A = []  # auxiliary matrix used in time integration
        self.outer_border_indices = {"top": [],
                                     "right": [],
                                     "bottom": [],
                                     "left": []
                                     }
        self.constrained_vertices = set()
        self.time_solver_type = None
        self.iteration = None

    def add_elem(self, elem: Element):
        if (max(*elem.node_ind) >= self.n_nodes) or (min(*elem.node_ind) < 0):
            raise ValueError("Element's node indices out of range")
        # TODO: check D matrix somehow
        if elem.rho <= 0 or elem.h <= 0:
            raise ValueError("Wrong element params:\n" + elem.to_string())
        self.elements.append(elem)

    def get_elem_midpoint(self, elem_ind: int) -> np.ndarray:
        """Returns middle point of element"""
        indices = np.array(self.elements[elem_ind].node_ind)
        vertex_coords = np.vstack((self.x_0[indices], self.y_0[indices]))
        return vertex_coords.mean(axis=1)

    def get_matplotlib_polygons(self) -> [Polygon]:
        polys = []
        for elem in self.elements:
            vertices = []
            for i in elem.node_ind:
                vertices.append([self.x_0[i], self.y_0[i]])
            polys.append(Polygon(vertices, closed=True, fill=False, edgecolor='b'))
        return polys

    def dump_vtk_grid(self, path: str):
        self.set_sigma()
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
            delta = np.array([[1.0, self.x_0[i], self.y_0[i]],
                              [1.0, self.x_0[j], self.y_0[j]],
                              [1.0, self.x_0[k], self.y_0[k]]])
            elem.S = np.linalg.det(delta)
            if elem.S < 0.0:
                elem.node_ind = elem.node_ind[::-1]  # if for some reason they are in clockwise order instead
                elem.S *= -1.0  # of counter-clockwise, change order and fix area

    def get_element_B(self, elem: Element):
        B = np.zeros([6, 9])
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            I, J, K = elem.node_ind[i], elem.node_ind[j], elem.node_ind[k]  # indices in external massive
            # TODO: omg this is so ugly and non-Python, need to find better solution
            beta = -(self.y_0[K] - self.y_0[J]) / elem.S
            gamma = (self.x_0[K] - self.x_0[J]) / elem.S
            B_1 = np.array([[beta, 0, 0],  # see eq 1.2.4
                            [0, gamma, 0],
                            [0, 0, 0]])
            B_2 = np.array([[gamma, beta, 0],
                            [0, 0, beta],
                            [0, 0, gamma]])  # assuming DB was already counted
            B[:, 3 * i:3 * (i + 1)] = np.row_stack((B_1, B_2))
        return B

    def set_DBmatrix(self):
        """"Calculates matrix linking stress and displacement for each element"""
        for elem in self.elements:
            D = elem.D
            B = self.get_element_B(elem)
            elem.DB = D @ B

    def assemble_K(self):  # K = 0.5Sh*B.T @(DB) see eq 1.4.1
        """"Calculates global stiffness matrix for entire grid; used in both static and dynamic problems"""
        for elem in self.elements:
            K_e = self.get_element_B(elem).T @ elem.DB
            K_e *= 0.5 * elem.S * elem.h  # local stiffness obtained
            for i in range(3):
                for j in range(3):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    self.K[3 * I:3 * (I + 1), 3 * J:3 * (J + 1)] += K_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]

    def assemble_f(self):
        """"Calculates nodal force vector from elements' distributed force vectors"""
        self.f.fill(0.0)
        for elem in self.elements:
            f_elem = elem.S * elem.h * elem.b(self.t) / 6.0
            for i in range(3):
                I = elem.node_ind[i]
                if I not in self.constrained_vertices:
                    self.f[3 * I:3 * (I + 1)] -= f_elem

    def assemble_M(self):
        """"Assembles mass matrix used in dynamic problem"""
        for elem in self.elements:
            I, J, K = elem.node_ind

            x = self.x_0[[I, J, K]]
            y = self.y_0[[I, J, K]]
            x -= np.average(x)
            y -= np.average(y)  # switching to barycenter cords

            A = x.T @ x
            B = y.T @ y
            C = x.T @ y  # coefs from mass matrix eq

            beta = -(np.roll(y, 1) - np.roll(y, 2)) / elem.S  # see eq 1.1.6
            gamma = (np.roll(x, 1) - np.roll(x, 2)) / elem.S  # x_k = np.roll(x,1)
            alpha = (np.roll(x, 2) * np.roll(y, 1) - np.roll(x, 1) * np.roll(y, 2)) / elem.S  # x_j = np.roll(x,2)
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
                    coef = 12.0 * alpha[i] * alpha[j]  # see eq 1.6.3
                    coef += A * beta[i] * beta[j]
                    coef += B * gamma[i] * gamma[j]
                    coef += C * (beta[i] * gamma[j] + gamma[i] * beta[j])
                    coef /= 2.0
                    M_ij = np.eye(3) * coef  # + elem.h**2*(V[i].T@V[j])
                    M_ij *= elem.rho * elem.S * elem.h / 12.0  # local submatrix assembled,

                    self.M[3 * I:3 * (I + 1), 3 * J:3 * (J + 1)] += M_ij  # mapping to global

    def constrain_velocity(self, node_index: int, velocity: np.ndarray):
        """Applies constraint a_t = velocity = const in poin node_index"""
        # TODO: verification
        self.constrained_vertices.add(node_index)
        self.a_t[3 * node_index: 3 * (node_index + 1)] = velocity
        self.K[3 * node_index: 3 * (node_index + 1), :] = 0
        self.M[3 * node_index: 3 * (node_index + 1), :] = 0
        self.M[3 * node_index: 3 * (node_index + 1), 3 * node_index: 3 * (node_index + 1)] = np.eye(3)
        self.f[3 * node_index: 3 * (node_index + 1)] = np.zeros(3)

    def apply_velocity_constraints(self, locs: list, vel: np.ndarray):
        """Applies constraint da/dt = vel at all points specified in locs array"""
        sides = ["top", "bottom", "left", "right"]
        while len(locs) > 0:
            item = locs.pop()
            if item == "border":
                locs += sides
                continue
            locs[:] = (it for it in locs if it != item)  # remove duplicates of item

            if item in sides:
                for k in self.outer_border_indices[item]:
                    self.constrain_velocity(k, vel)
            else:
                loc = np.array(item)
                if loc.shape[0] != 2:
                    raise ValueError(str(item))
                k = self.get_closest_vertex_index(loc)
                self.constrain_velocity(k, vel)

    def get_closest_vertex_index(self, loc: np.ndarray):
        if loc.shape[0] != 2:
            raise ValueError(f"loc must be a 2-d numpy ndarray [x, y]. Got {loc}")
        x, y = loc[0], loc[1]
        dist = (self.x_0 - x) ** 2 + (self.y_0 - y) ** 2
        return np.argmin(dist)

    def estimate_tau(self):
        """"Estimates time step that tau < diam/c, c ~ sqrt(E/rho) (that everything converges"""
        def tau(e: Element):
            E_eff = np.diag(e.D)[:3].max() # TODO: find actual formula for wave speed
            return np.sqrt(e.S * e.rho / E_eff)  # this may be wrong in counting both diameter

        tau_max = 0.5 * min([tau(e) for e in self.elements])  # and c, but i want to test the thing already
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
            a_e = np.concatenate([self.a[3 * I:3 * (I + 1)],
                                  self.a[3 * J:3 * (J + 1)],
                                  self.a[3 * K:3 * (K + 1)]])  # getting displacement vector from global array
            elem.sigma = elem.DB @ a_e

    def set_Newmark_params(self, beta_1: np.float64, beta_2: np.float64, tau: np.float64):
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.A = sp.linalg.inv(self.M + 0.5 * beta_2 * tau ** 2 * self.K)

    def iteration_Newmark(self):
        self.assemble_f()

        a_est = self.a + self.tau * self.a_t + 0.5 * (1 - self.beta_2) * self.tau ** 2 * self.a_tt
        v_est = self.a_t + (1 - self.beta_1) * self.tau * self.a_tt

        a_tt_next = self.A.dot((-self.K.dot(a_est) - self.f))  # actually we need f_n+1 here

        self.a_tt = a_tt_next
        self.a_t = v_est + self.beta_1 * self.tau * a_tt_next
        self.a = a_est + 0.5 * self.beta_2 * self.tau ** 2 * a_tt_next
        self.t += self.tau

    def set_Newmark_noinverse(self, beta_1: np.float64, beta_2: np.float64, tau: np.float64):
        """Sets params for Newmark iteration without preliminary inverting A matrix"""
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.A = self.M + 0.5 * beta_2 * tau ** 2 * self.K

    def iteration_Newmark_noinverse(self):
        """"Newmark iteration not using preliminary inverted matrix A (solving at each step)"""
        self.assemble_f()

        a_est = self.a + self.tau * self.a_t + 0.5 * (1 - self.beta_2) * self.tau ** 2 * self.a_tt
        v_est = self.a_t + (1 - self.beta_1) * self.tau * self.a_tt

        a_tt_next = sp.linalg.spsolve(self.A, (-self.K.dot(a_est) - self.f))  # actually we need f_n+1 here

        self.a_tt = a_tt_next
        self.a_t = v_est + self.beta_1 * self.tau * a_tt_next
        self.a = a_est + 0.5 * self.beta_2 * self.tau ** 2 * a_tt_next
        self.t += self.tau

    def set_Newmark_factorized(self, beta_1: np.float64, beta_2: np.float64, tau: np.float64):
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.A = scipy.sparse.linalg.splu(self.M + 0.5 * beta_2 * tau ** 2 * self.K)

    def iteration_factorized(self):
        self.assemble_f()
        a_est = self.a + self.tau * self.a_t + 0.5 * (1 - self.beta_2) * self.tau ** 2 * self.a_tt
        v_est = self.a_t + (1 - self.beta_1) * self.tau * self.a_tt

        a_tt_next = self.A.solve(-self.K.dot(a_est) - self.f)  # actually we need f_n+1 here

        self.a_tt = a_tt_next
        self.a_t = v_est + self.beta_1 * self.tau * a_tt_next
        self.a = a_est + 0.5 * self.beta_2 * self.tau ** 2 * a_tt_next
        self.t += self.tau

    def set_time_integration_mode(self, tau: float, tp: str or None = "no_inverse"):
        if tp is None or tp == "no_inverse":
            self.set_Newmark_noinverse(0.5, 0.5, tau)
            self.iteration = self.iteration_Newmark_noinverse
        elif tp == "inverse":
            self.set_Newmark_params(0.5, 0.5, tau)
            self.iteration = self.iteration_Newmark
        elif tp == "factorized":
            self.set_Newmark_factorized(0.5, 0.5, tau)
            self.iteration = self.iteration_factorized

    def set_loading(self, load_expr: List[sympy.Expr]):
        if len(load_expr) != 3:
            raise ValueError(f"Load function must specify 3 components, got {len(load_expr)}")
        for i, elem in enumerate(self.elements):
            mp = self.get_elem_midpoint(i)
            x, y = mp[0], mp[1]
            load_vec = [comp.subs({'x': x, 'y': y}) for comp in load_expr]
            func = sympy.lambdify('t', sympy.Matrix(load_vec),
                                  [{'ImmutableDenseMatrix': (lambda z: np.array(z)[:, 0])}, 'numpy'])
            elem.b = func

    def get_radial_distribution(self, f, center_cord: np.ndarray):
        """"
        Counts radial distribution of functions [f_1(u, v), ...] over vertices
        Args:
        f - a function that takes u, v (np arrays of shape (3,)) and returns
        list of [f_1, f_2, f_3 ...]
        Returns: array [r_i, f_1(u(_i, v_i), ...]"""

        def dist(i: int):
            cord = np.array([self.x_0[i], self.y_0[i]])
            return np.linalg.norm(cord - center_cord, ord=2)

        res = [[dist(i)] + f(self.a[3 * i:3 * (i + 1)], self.a_t[3 * i:3 * (i + 1)]) for i in range(self.n_nodes)]
        return np.array(res).T

    def discard_displacement(self):
        self.a = np.zeros([3 * self.n_nodes])  # vector of nodal displacements
        self.a_t = np.zeros([3 * self.n_nodes])  # vector of nodal velocities
        self.a_tt = np.zeros([3 * self.n_nodes])  # vector of nodal accelerations

    def serialize(self, pickle_file: str):
        # discarding lambdas to serialize grid
        for e in self.elements:
            e.b = None
        with open(pickle_file, 'wb') as ofile:
            pickle.dump(self, ofile, protocol=pickle.HIGHEST_PROTOCOL)


def get_isotropic_elastic_tensor(E: np.float64, nu: np.float64) -> np.ndarray:
    """" """
    D_2 = np.identity(3) * (1 - 2.0 * nu)
    D_1 = np.full([3, 3], nu) + D_2
    D_2 /= 2.0
    D = E / (1 + nu) / (1 - 2.0 * nu)
    D_1 *= D
    D_2 *= D
    return np.vstack((np.hstack((D_1, np.zeros_like(D_1))),
                      np.hstack((np.zeros_like(D_2), D_2))))


def generate_perforated_grid(X: np.float64, Y: np.float64, n_x: int, n_y: int,
                             step_x: int, step_y: int,
                             h: np.float64, rho: np.float64,
                             D: np.ndarray = None,
                             E: np.float64 = None, nu: np.float64 = None) -> Grid:
    if D is None and (E is None or nu is None):
        raise ValueError("Either a full elastic tensor D or elastic constants (E, nu)"
                         "must be specified")

    if D is None:
        D = get_isotropic_elastic_tensor(E, nu)

    res = Grid((n_x + 1) * (n_y + 1))

    def index(i, j):
        return (n_x + 1) * i + j

    dx = X / n_x
    dy = Y / n_y

    for i in range(n_y + 1):
        for j in range(n_x + 1):
            k = index(i, j)
            res.x_0[k], res.y_0[k] = dx * j, dy * i
            # adding indices to border array
            if j == 0:
                res.outer_border_indices["left"].append(k)
            if j == n_x:
                res.outer_border_indices["right"].append(k)
            if i == 0:
                res.outer_border_indices["bottom"].append(k)
            if i == n_y:
                res.outer_border_indices["top"].append(k)

    for i in range(n_y):
        for j in range(n_x):
            if (j + 1) % step_x == 0 and (i + 1) % step_y == 0 and i != n_y - 1 and j != n_x - 1:
                continue
            res.elements.append(Element((index(i, j), index(i, j + 1), index(i + 1, j + 1)), D, h, rho))
            res.elements.append(Element((index(i, j), index(i + 1, j + 1), index(i + 1, j)), D, h, rho))
    return res


def generate_uniform_grid(X: np.float64, Y: np.float64, n_x: int, n_y: int,
                          h: np.float64, rho: np.float64,
                          D=None, E=None, nu=None) -> Grid:
    return generate_perforated_grid(X, Y, n_x, n_y, n_x + 1, n_y + 1, h, rho, D=D, E=E, nu=nu)


def store_random_grid(path: str, n_x: int, n_y: int, n_inner: int,
                      E: np.float64, nu: np.float64,
                      h: np.float64, rho: np.float64, X: np.float64 = 1.0, Y: np.float64 = 1.0,
                      disp: np.float64 = 0.1):
    """
    Creates a square grid of size X * Y. It has n_x + 1 or n_y + 1  vertices on sides and n_vert in total
    The routine assembles M and K matrices and stores everything in pickle file specified by path
    """

    x_border = np.linspace(0, X, n_x + 1, endpoint=True)
    y_border = np.linspace(0, Y, n_y + 1, endpoint=True)

    cords = np.vstack((x_border, np.full_like(x_border, 0.0)))  # lower border
    cords = np.append(cords, np.vstack((x_border, np.full_like(x_border, Y))), axis=1)  # upper border

    y_border = y_border[1:-1]

    cords = np.append(cords, np.vstack((np.full_like(y_border, 0.0), y_border)), axis=1)  # left border
    cords = np.append(cords, np.vstack((np.full_like(y_border, X), y_border)), axis=1)  # right border

    x_inner = np.random.normal(X / 2, disp, n_inner * 2)
    y_inner = np.random.normal(Y / 2, disp, n_inner * 2)

    index_x = np.logical_and(x_inner > 0, x_inner < X)
    index_y = np.logical_and(y_inner > 0, y_inner < Y)

    index = np.logical_and(index_x, index_y)
    inner_cords = np.vstack((x_inner[index], y_inner[index]))[:, :n_inner]

    if inner_cords.shape[1] < n_inner:
        warnings.warn("Not enough inner points generated", RuntimeWarning)

    cords = np.append(cords, inner_cords, axis=1)

    n_vertex = cords.shape[1]
    res = Grid(n_vertex)
    res.x_0 += cords[0]
    res.y_0 += cords[1]

    triangulation = Delaunay(cords.T)

    for triangle in triangulation.simplices:
        triangle = list(map(int, triangle))
        node_ind = (triangle[0], triangle[1], triangle[2])
        el = Element(node_ind, E, nu, h, rho)
        res.elements.append(el)

    res.ready()
    res.K = res.K.tocsc()
    res.M = res.M.tocsc()

    with open(path, 'wb') as ofile:
        pickle.dump(res, ofile, protocol=pickle.HIGHEST_PROTOCOL)

#    /|
#   / +============================|
#  X       To be continued   | \ /||
#   \ +============================|
#    \|
