import twa_model as twa
import matplotlib.pyplot as plt

k2eV = 4.55633e-6
parameters = {'cavity freq': 1600 * k2eV,  # (cm-1) -> (A.U.)
              'vibration freq': 1600 * k2eV,  # (cm-1) -> (A.U.)
              'Huang-Rhys': 1,
              'coupling': 0.04e-3,  # (A.U)
              'final time': 10,  # (ps)
              'time steps': 4000,
              'field size': 500,
              'bath size': 500,
              'res range': 878 * k2eV,  # (cm-1) -> (A.U.)
              'vibration lifetime': 2,  # (ps)
              'Q': 1600  #  Cavity quality factor
              }

polEn, polStates, siteEn = twa.polariton_modes(parameters)
polDispl = twa.pol_displacement(parameters, polEn, polStates, siteEn)
nCav = twa.oc_dynamics(parameters, polEn, polStates, siteEn, polDispl, subsystem='cavity')
nVib = twa.oc_dynamics(parameters, polEn, polStates, siteEn, polDispl, subsystem='vibration')
nField = twa.oc_dynamics(parameters, polEn, polStates, siteEn, polDispl, subsystem='field')
nBath = twa.oc_dynamics(parameters, polEn, polStates, siteEn, polDispl, subsystem='bath')

plt.plot(nCav[0], nCav[1], label='cavity', color='k')
plt.plot(nVib[0], nVib[1], label='vibration', color='tab:blue')
plt.plot(nField[0], nField[1], label='external field', color='tab:orange')
plt.plot(nBath[0], nBath[1], label='non-radiative bath', color='tab:green')
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
legend = plt.legend(title='$x = $', loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, fontsize=16)
legend.get_title().set_fontsize('20')
legend.get_frame().set_edgecolor('black')
