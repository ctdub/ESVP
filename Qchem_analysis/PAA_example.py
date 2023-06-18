import numpy as np
import matplotlib.pyplot as plt
import QC_analysis as QC


def normalize(arr):
    max = np.amax(arr)
    normed = arr/max
    return normed


# creates an object that will contain all of the information from the QChem output file
dat = QC.ImportData(['PAA_geoopt_freq.out', 'PAA_force.out'])
dat.import_freq_data(file_number=0, mw_hessian=True)
dat.import_force_data(file_number=1)
dat.diag_mw_hessian()
dat.calc_HR()

E0 = 345  # nm
electric_energy = 45.5640 / E0
cycles = 1600
ps2AU = 1/2.4189e-5
totalTime = 0.65  # ps
totalSteps = 20000
# totalTimeAU = totalTime * ps2AU
totalTimeAU = cycles * 2 * np.pi / electric_energy
timeRangeAU = np.linspace(0, totalTimeAU, totalSteps)
T = 200  # Kelvin
inhomBroadening = 200  # cm-1
freqAxis, spec = QC.abs_spec(E0, dat.mw_frequencies, dat.hr, T, inhomBroadening, timeRangeAU)

plt.figure()

plt.figure()
plt.plot(freqAxis, normalize(spec), 'r')
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
