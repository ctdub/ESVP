import numpy as np
from numpy import linalg as LA

k2eV = 4.55633e-6
c = 2.998e10  # speed of light in cm/s


def polariton_modes(pr):
    dF = pr['res range'] / 2
    field_en = np.linspace(pr['cavity freq'] - dF, pr['cavity freq'] + dF, pr['field size']) ** 2
    bath_en = np.linspace(pr['vibration freq'] - dF, pr['vibration freq'] + dF, pr['bath size']) ** 2

    fDamp = 1 / (2 * np.pi * pr['Q'])
    kappa = np.sqrt(fDamp * np.sqrt(field_en) * 2 * dF / pr['field size'])
    fieldG = 2 * np.sqrt(pr['cavity freq'] * np.sqrt(field_en)) * kappa
    fieldG = np.append(fieldG, np.zeros(pr['bath size']))

    omegaCHz = 2 * np.pi * c * pr['cavity freq'] / k2eV
    QBath = omegaCHz * pr['vibration lifetime'] * 1e-12
    bDamp = 1 / (2 * np.pi * QBath)
    chi = np.sqrt(bDamp * np.sqrt(bath_en) * 2 * dF / pr['bath size'])
    bathG = 2 * np.sqrt(pr['vibration freq'] * np.sqrt(bath_en)) * chi

    sysG = 2 * np.sqrt(pr['cavity freq'] * pr['vibration freq']) * pr['coupling']
    cavG = np.append(sysG, fieldG)

    energies = np.append([pr['cavity freq'] ** 2, pr['vibration freq'] ** 2], np.append(field_en, bath_en))
    hessian = np.diag(energies)
    hessian[1:, 0] = cavG
    hessian[0, 1:] = cavG
    hessian[2 + pr['field size']:, 1] = bathG
    hessian[1, 2 + pr['field size']:] = bathG
    v, u = LA.eigh(hessian)
    return v, u, energies


def pol_displacement(pr, v, u, energies):
    totalSize = v.size
    displ = np.zeros(totalSize)
    displ[1] = np.sqrt(2 * pr['Huang-Rhys'] / pr['vibration freq'])  # displacement along the vibrational mode
    return u.T.dot(energies * displ) / v


def oc_dynamics(pr, v, u, energies, polDispl, subsystem='cavity'):
    time = np.linspace(0, pr['final time'], pr['time steps'])
    timeAU = time / 2.41889E-5
    totalSize = v.size
    omega_t = np.tile(timeAU, (totalSize, 1)).T * np.sqrt(v)

    subSize = 1
    if subsystem == 'cavity':
        start = 0
        stop = 1
    if subsystem == 'vibration':
        start = 1
        stop = 2
    if subsystem == 'field':
        start = 2
        stop = pr['field size'] + 2
        subSize = pr['field size']
    if subsystem == 'bath':
        start = pr['bath size'] + 2
        stop = pr['bath size'] + pr['field size'] + 2
        subSize = pr['bath size']
    if subsystem not in {'cavity', 'vibration', 'field', 'bath'}:
        if not isinstance(subsystem, str):
            raise ValueError('invalid variable type for subsystem')
        raise ValueError('invalid subsystem name: ' + subsystem + '\n Subsystem can be cavity, vibration, field, or bath')
    if subSize == 0:
        proj = np.zeros(v.size)[np.newaxis, :]
        subEn = [1]
    else:
        proj = u[start:stop, :]
        subEn = energies[start:stop]

    P = np.einsum('ij,kj', proj, polDispl * np.sqrt(v) * np.sin(omega_t)).T
    Q = np.einsum('ij,kj', proj, polDispl * (np.sin(omega_t / 2) ** 2)).T

    n_t = np.sum(1 / (2 * np.sqrt(subEn)) * np.square(P) + 2 * np.sqrt(subEn) * np.square(Q), axis=1)
    nBeta = np.sum(
        1 / np.sqrt(subEn) * np.square(proj).dot(np.sqrt(v)) + np.sqrt(subEn) * np.square(proj).dot(1 / np.sqrt(v)))

    n_t2 = 0
    if subsystem == 'vibration':
        n_t2 = - 2 * np.sqrt(2 * pr['Huang-Rhys'] * pr['vibration freq']) * np.sum(Q, axis=1) + pr['Huang-Rhys']
    return [time, n_t + n_t2 + 0.25 * nBeta - 0.5 * subSize]
