import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import QC_analysis as QC

datGS = QC.ImportData(['PAA_geoopt_freq.out', 'PAA_force.out'])
datGS.import_freq_data(file_number=0, mw_hessian=True)
datGS.import_force_data(file_number=1)
datGS.diag_mw_hessian()
datGS.calc_HR()

datES = QC.ImportData(['PAA_esVib_gsGeo.out'])
datES.import_freq_data(file_number=0, mw_hessian=True)
datES.diag_mw_hessian()

prob = datGS.norms_from_mw_hess.T.dot(datES.norms_from_mw_hess) ** 2
maxIndx = np.argmax(prob[6:, 6:], 1)
IR_gs = datES.ir_dipole[maxIndx, :]
normIR_gs = np.sum(IR_gs**2, 1)**0.5
sThresh = np.argwhere(datGS.hr > 0.012)[:, 0]
plt.bar(datGS.frequency[sThresh], datGS.hr[sThresh], 40, color='#3FADFE')
plt.bar(datGS.frequency[sThresh], normIR_gs[sThresh], 40, bottom=datGS.hr[sThresh], color='#094948')
