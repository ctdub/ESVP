import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import QC_analysis as QC


def normalize(arr):
    max = np.amax(arr)
    normed = arr/max
    return normed


ps2AU = 1/2.4189e-5

# creates an object that will contain all of the information from the QChem output file
dat = QC.ImportData(['PAA_geoopt_freq.out', 'PAA_force.out'])
dat.import_freq_data(file_number=0, mw_hessian=True)
dat.import_force_data(file_number=1)
dat.diag_mw_hessian()
dat.calc_HR()

E0 = 345  # nm
totalTime = 1  # ps
totalSteps = 20000
timeRangeAU = np.linspace(0, totalTime * ps2AU, totalSteps)
T = 300  # Kelvin
inhomBroadening = 200  # cm-1
freqAxis, spec_hot = QC.abs_spec(E0, dat.mw_frequencies, dat.hr, T, inhomBroadening, timeRangeAU)
T = 10
inhomBroadening = 10  # cm-1
freqAxis, spec_frozen = QC.abs_spec(E0, dat.mw_frequencies, dat.hr, T, inhomBroadening, timeRangeAU)

plt.figure()
exp_abs = np.loadtxt('PAA_abs.txt')
plt.plot(1e7/exp_abs[61:,0], normalize(exp_abs[61:, 1]), color='k')
plt.plot(freqAxis, normalize(spec_hot), 'r')
plt.plot(freqAxis, normalize(spec_frozen), 'tab:blue')
plt.ylabel(r'Intensity', fontsize=24)
plt.xlabel(r'Wavenumbers / $\mathrm{cm}^{-1}$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.locator_params(axis='x', nbins=6)
plt.xlim(27000, 38000)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(width=2)
plt.gca().yaxis.set_tick_params(width=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()
