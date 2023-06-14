import twa_model as twa
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

k2eV = 4.55633e-6
param = {'cavity freq': 1600 * k2eV,
         'vibration freq': 1600 * k2eV,
         'Huang-Rhys': 0.75,
         'coupling': 0.5e-3,
         'final time': 0.45,
         'time steps': 1000
         }

bathParam = {'field size': 5,
             'bath size': 5,
             'max bath freq': 878 * k2eV}

hessian = np.zeros((2, 2))
hessian[0, 0] = param['cavity freq'] ** 2
hessian[1, 1] = param['vibration freq'] ** 2
G = 2 * np.sqrt(param['cavity freq'] * param['vibration freq']) * param['coupling']
hessian[1, 0] = G
hessian[0, 1] = G

v, u = LA.eigh(hessian)

energies = np.array([param['cavity freq'], param['vibration freq']]) ** 2
displ = np.zeros(2)
displ[1] = np.sqrt(2 * param['Huang-Rhys'] / param['vibration freq'])  # displacements along vibrational mode
polDispl = u.T.dot(energies * displ) / v

time = np.linspace(0, param['final time'], param['time steps'])
timeAU = time / 2.41889E-5
t1 = np.sum(u[0, :] * polDispl * np.sqrt(v) * np.sin(np.tile(timeAU, (2, 1)).T * np.sqrt(v)), axis=1)
t2 = np.sum(u[0, :] * polDispl * (np.sin(np.tile(timeAU, (2, 1)).T * np.sqrt(v) / 2) ** 2), axis=1)

ncav_t = 1 / (2 * param['cavity freq']) * np.square(t1) + 2 * param['cavity freq'] * np.square(t2)
ncavBeta = 1 / param['cavity freq'] * np.square(u[0, :]).dot(np.sqrt(v)) + param['cavity freq'] * np.square(u[0, :]).dot(1 / np.sqrt(v))
ncav = ncav_t + 0.25 * ncavBeta - 0.5


ncav1 = 1 / (2 * param['cavity freq']) * np.square(t1) + 2 * param['cavity freq'] * np.square(t2)
ncavBeta = 1 / param['cavity freq'] * np.square(u[0, :]).dot(np.sqrt(v)) + param['cavity freq'] * np.square(u[0, :]).dot(1/np.sqrt(v))
ncav2 = ncav1 + 0.25 * ncavBeta - 0.5

tv1 = np.sum(u[1, :] * polDispl * np.sqrt(v) * np.sin(np.tile(timeAU, (2, 1)).T * np.sqrt(v)), axis=1)
tv2 = np.sum(u[1, :] * polDispl * (np.sin(np.tile(timeAU, (2, 1)).T * np.sqrt(v) / 2) ** 2), axis=1)

nvib_t = 1 / (2 * param['vibration freq']) * np.square(tv1) + 2 * param['vibration freq'] * np.square(tv2)
nvib_t2 = - 2 * param['vibration freq'] * tv2
nvibBeta = 1 / param['vibration freq'] * np.square(u[1, :]).dot(np.sqrt(v)) + param['vibration freq'] * np.square(u[1, :]).dot(1 / np.sqrt(v))
nvib = nvib_t2 + 0.25 * nvibBeta - 0.5 + param['Huang-Rhys']
# 4 * param['Huang-Rhys'] * param['vibration freq'] ** 2 *
# plt.plot(time, ncav, color='r')
# plt.plot(time, ncav2, 'b:')
plt.plot(time, nvib_t, 'g')
plt.plot(time, nvib_t2, 'r')
plt.plot(time, nvib_t + nvib_t2 + 0.25 * nvibBeta - 0.5 + param['Huang-Rhys'], 'k')
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('time / ps', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(r'$\langle N_{x} \rangle$', fontsize=20)
plt.xticks(fontsize=16)
plt.gca().xaxis.set_tick_params(width=2)
plt.gca().yaxis.set_tick_params(width=2)
# legend = plt.legend(title='$x = $', loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, fontsize=16)
# legend.get_title().set_fontsize('20')
# legend.get_frame().set_edgecolor('black')
