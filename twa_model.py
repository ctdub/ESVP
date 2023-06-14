import numpy as np
from numpy import linalg as LA


def polariton_modes(param):
    hessian = np.zeros((2, 2))
    hessian[0, 0] = param['cavity freq'] ** 2
    hessian[1, 1] = param['vibration freq'] ** 2
    G = 2 * np.sqrt(param['cavity freq'] * param['vibration freq']) * param['coupling']
    hessian[1, 0] = G
    hessian[0, 1] = G
    return LA.eigh(hessian)


def occ_dynamics(param, v, u):
    energies = np.array([param['cavity freq'], param['vibration freq']]) ** 2
    displ = np.zeros(2)
    displ[1] = np.sqrt(2 * param['Huang-Rhys'] / param['vibration freq'])  # displacements along vibrational mode
    polDispl = u.T.dot(energies * displ) / v

    time = np.linspace(0, param['final time'], param['time steps'])
    timeAU = time / 2.41889E-5
    omega_t = np.tile(timeAU, (2, 1)).T * np.sqrt(v)
    t1 = np.sum(u[0, :] * polDispl * np.sqrt(v) * np.sin(omega_t), axis=1)
    t2 = np.sum(u[0, :] * polDispl * (np.sin(omega_t) ** 2), axis=1)

    ncav_t = 1 / (2 * param['cavity freq']) * np.square(t1) + 2 * param['cavity freq'] * np.square(t2)
    ncavBeta = 1 / param['cavity freq'] * np.square(u[0, :]).dot(np.sqrt(v)) + param['cavity freq'] * np.square(
        u[0, :]).dot(1 / np.sqrt(v))
    return ncav_t + 0.25 * ncavBeta - 0.5

