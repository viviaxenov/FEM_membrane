import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt


import membrane as mb
from config_loader import run_command

import os
import sys
import functools
import pickle
import time


class IdentificationProblem:
    def get_Ke_basis(g: mb.Grid):
        Ke_basis = []
        counter = 0
        for i in range(6):
            for j in range(6):
                if j < i:
                    continue
                De = np.zeros((6,6), dtype=np.float64)
                De[i, j] = 1
                De[j, i] = 1
                g.assemble_K(De)
                #TODO generate files with meaningful names for constraints
                g.apply_velocity_constraints(['constraint'], np.array([.0, .0, .0,])) 
                Ke_basis.append(g.K.copy())
                counter += 1
                print(f'Basis matrix for {counter:-1d}-st component assembled', flush=True)
        return Ke_basis

    def get_K(Ke_basis: list, D_vec: np.array):
        K = sp.dok_matrix(Ke_basis[0].shape)
        for i in range(21):
            K += D_vec[i]*Ke_basis[i]
        return K.tocsc()

    def Dmatrix_to_vector(D):
        D_vec = np.zeros(21)
        counter = 0
        for i in range(6):
            for j in range(6):
                if j < i:
                    continue
                D_vec[counter] = D[i, j]
                counter += 1
        return D_vec

    def Dvector_to_matrix(D_vec):
        D = np.zeros((6,6))
        counter = 0
        for i in range(6):
            for j in range(6):
                if j < i:
                    continue
                D[i, j] = D_vec[counter]
                D[j, i] = D_vec[counter]
                counter += 1
        return D

    def __init__(self, model: mb.Grid, gamma=1.4, D_ref=None):
        self.D = model.D.copy() # interpret it as an initial guess
        self.Dv = IdentificationProblem.Dmatrix_to_vector(self.D)
        self.Ke_basis = IdentificationProblem.get_Ke_basis(model)
        self.K = IdentificationProblem.get_K(self.Ke_basis, self.Dv)

        self.M = model.M
        self.gamma = gamma
        self.f = model.f.copy()
        self.D_ref = D_ref
        self.afc_residual = np.inf
        self.D_residual = np.inf

        self.grad = np.zeros(21)

        self.default_linear_solver = functools.partial(sp.linalg.bicgstab, tol=1e-8, atol=1e-13)

        self.omegas = None
        self.u_ref = None
        self.u = None

    def add_reference(self, omegas: np.array, afc_ref: np.array):
        self.omegas = omegas.copy()
        assert afc_ref.shape == (omegas.shape[0], self.f.shape[0])
        self.u_ref = afc_ref.copy()
        self.u = np.zeros_like(afc_ref)

    def eval_u(self):
        self.u.fill(0)
        for idx, w in enumerate(self.omegas):
            K_eff = self.K + 1j*self.gamma*w*self.M - w**2*self.M
            u, status = self.default_linear_solver(K_eff, self.f)
            self.u[idx, :] = u

    def eval_residuals(self):
        self.afc_residual = np.linalg.norm(self.u_ref - self.u)
        if self.D_ref is not None:
            self.D = IdentificationProblem.Dvector_to_matrix(self.Dv)
            self.D_residual = np.linalg.norm(self.D - self.D_ref)


    def eval_gradient(self):
        self.grad.fill(0.)
        for idx, w in enumerate(self.omegas):
            K_eff = self.K + 1j*gamma*w*self.M - w**2*self.M
            u = self.u[idx, :] 
            for j in range(21):
                partial, status = self.default_linear_solver(K_eff, -self.Ke_basis[j]@u)
                vp = np.vdot(u - self.u_ref[idx, :], partial).real
                self.grad[j] += vp

    def gradient_descent_step(self, grad_step=.1):
        self.eval_gradient()
        # gradient descent step
        self.Dv -= grad_step*self.grad

    def optimize_gd(self, grad_step=.1, max_steps=1000, rtol=1e-8, verbose=False):
        for step in range(max_steps):
            self.K = IdentificationProblem.get_K(self.Ke_basis, self.Dv)
            self.eval_u()
            self.eval_residuals()

            if self.afc_residual <= np.linalg.norm(self.u_ref)*rtol:
                self.D = IdentificationProblem.Dvector_to_matrix(self.Dv)
                print(f'Optimization converged at step # {step + 1:d}')
                return
            start = time.perf_counter()
            self.gradient_descent_step(grad_step)
            t_exec = time.perf_counter() - start
            if verbose:
                with np.printoptions(precision=2,linewidth=1000):
                    print(self.grad)
                print(f'{step:-5d} step, {self.afc_residual:-.10e} residual afc, {self.D_residual:-.10e} residual D, {t_exec:.1f}s exec time')
        
        self.D = IdentificationProblem.Dvector_to_matrix(self.Dv)
        print(f'Relative tolerance {rtol:.1e} not achieved after {step:d} iterations')

if __name__ == '__main__':
    from parse_inp import parse_inp_file
    pickle_path = '../pickles/id'


    def get_afc(K, M, f, gamma, omegas):
        n_freqs = omegas.shape[0]
        us = np.zeros((n_freqs, f.shape[0]), dtype=np.complex128)
        C = gamma*M
        for i in range(n_freqs):
            w = omegas[i]
            K_eff = K + 1j*w*C - w**2*M
            u0, res = sp.linalg.bicgstab(K_eff, f, tol=1e-8, atol=1e-13)
            us[i, :] = u0
        return us


    def adhoc_assemble_f(grid: mb.Grid):
        grid.f.fill(0.0)
        for ind in grid.element_groups['load']:
            elem = grid.elements[ind]
            f_elem = elem.S * elem.h * 1e8 * np.array([0.,0.,1.]) / 6.0
            for i in range(3):
                I = elem.node_ind[i]
                grid.f[3*I:3*(I + 1)] -= f_elem
        return

    if '-r' in sys.argv[1:] or not os.path.exists(pickle_path):

        model = run_command("grid ../configs/identification/model.json")
        adhoc_assemble_f(model)
        identification = IdentificationProblem(model, gamma=gamma, D_ref=reference.D)

        with open(pickle_path, 'wb') as ofile:
            pickle.dump(identification, ofile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_path, 'rb') as ifile:
            identification = pickle.load(ifile)
    
    reference = run_command("grid ../configs/identification/ref.json")
    adhoc_assemble_f(reference)
    gamma = 1.6

    omegas = np.linspace(110, 120, 51, endpoint=True)
    omegas *= 2*np.pi
    afc_ref = get_afc(reference.K, reference.M, reference.f, gamma, omegas) 
    n = 83 # test node
    identification.add_reference(omegas, afc_ref)
    identification.eval_u()
    afc_old = identification.u[:, 3*n + 2].copy()
    identification.optimize_gd(max_steps=1000, verbose=True, grad_step=1e12)

    plt.plot(omegas/2/np.pi, np.absolute(afc_ref[:, 3*n + 2]), label='Reference', marker='s')
    plt.plot(omegas/2/np.pi, np.absolute(afc_old), label='Initial guess', marker='o')
    plt.plot(omegas/2/np.pi, np.absolute(identification.u[:, 3*n + 2]), label='Optimized', marker='o')
    #label=f"({grid.x_0[n]:.1f}, {grid.y_0[n]:.1f})"
    #    plt.plot(omegas/2/np.pi, us[:, 3*n + 2].imag, label=f"Im(f)")
    plt.grid()
    plt.legend()
    plt.xlabel('Frequency, Hz')
    plt.ylabel("$\\|f\\|$")
    plt.show()
