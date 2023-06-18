import mmap
import numpy as np
from numpy import linalg as LA
import re
import math

# conversion
bohr2m = 0.529177249e-10
hartree2joule = 4.35974434e-18
speed_of_light = 299792458
avogadro = 6.0221413e+23
auforce2newton = 8.238726e-08
vib_constant = np.sqrt((avogadro * hartree2joule * 1000) / (bohr2m * bohr2m)) / (2 * math.pi * speed_of_light * 100)
kToAU = 4.5563323e-06


def split(strs, delimiter=r'\s+', datatype='utf-8'):
    if datatype == 'utf-8':
        np_strs = np.asarray(re.split(delimiter, strs.decode(datatype)))
    else:
        np_strs = np.asarray(re.split(delimiter, strs))
    np_strs = np_strs[np_strs != '']
    ii = 0
    for txt in np_strs:
        if not (txt.strip('-.').replace('.', '')).isdigit():
            np_strs = np.delete(np_strs, ii)
            ii -= 1
        ii += 1
    np_strs = np_strs.astype(float)
    return np_strs


def find_geometry(file_name):
    count_on = False  # turns on when line hits optimized geometry section
    geo = np.array([])  # molecular geometry in nuclear coordinates
    element_array = np.array([])
    num_nuc = 0  # Counts the number of nuclei
    nuc_counter = False  # Turns on when it hits the input geometry to count # of nuclei
    firstNucCount = True
    startGeo = True  # Tells if it is the 1st case of reading $molecule. Turns off after the 1st instance.
    stdMolCount = False  # Turned on while reading geometry from 1st $molecule section.

    # create memory map to file and read each bit line
    with open(file_name, mode='r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_f:
            for line in iter(mmap_f.readline, b''):
                if count_on and number > 0 and b'\n' == line:
                    break
                if stdMolCount and b'------' in line and number > 0:
                    stdMolCount = False
                    number = 0
                    nuc_counter = False
                if count_on or stdMolCount:
                    number += 1
                    if number > 0:
                        split_str = np.asarray(re.split(r'\s+', line.decode('utf-8')))
                        split_str = split_str[split_str != '']
                        element_array = np.concatenate(
                            (element_array, np.array([split_str[1]])))  # pull out the element
                        geo = np.concatenate((geo, np.delete(split(line), 0)), axis=0)
                        if nuc_counter:
                            num_nuc += 1

                #  If a geometry optimization is done this will pull the final optimized geometry and overwrite the
                #  geometry from $molecule
                if b'**  OPTIMIZATION CONVERGED  **' in line:
                    number = -4  # number of molecules. -4 to skip first 4 lines after **  OPTIMIZATION CONVERGED  **'
                    count_on = True
                    geo = np.array([])  # molecular geometry in nuclear coordinates
                    element_array = np.array([])

                #  This pulls the geometry from the user inputted "Standard Nuclear Orientation" section.
                #  This is overwritten if a geometry optimization is done.
                if b'Standard Nuclear Orientation (Angstroms)' in line and firstNucCount:
                    if startGeo:
                        number = -2  # number of molecules. -1 to skip first two lines after 'Standard Nuclear... '
                        stdMolCount = True
                        startGeo = False
                    if firstNucCount:
                        nuc_counter = True
                        firstNucCount = False
    size = num_nuc * 3  # 3N normal modes including trans and rot
    return num_nuc, size, geo, element_array


class ImportData:
    def __init__(self, file_name, geofilenum=0):
        self.my_file = []
        if isinstance(file_name, list):
            self.my_file.extend(file_name)
        else:
            self.my_file.append(file_name)
        self.num_atoms, self.norm_size, self.geometry, self.elements = find_geometry(self.my_file[geofilenum])
        self.frequency = np.zeros(self.norm_size - 6, dtype='g')
        self.hessian = np.array([])
        self.mw_hessian = np.array([])
        self.norm = np.zeros((self.norm_size, self.norm_size - 6))
        self.masses = np.zeros(self.num_atoms) # stores the mass of each atom indexed in the same order as the nuclear coordinates
        self.ground_force_cnst = np.zeros(self.norm_size - 6)
        self.red_mass = np.zeros(self.norm_size - 6)
        self.ir_intensity = np.zeros(self.norm_size - 6)
        self.force = np.zeros(self.norm_size)  # actually gradient of excited state energy, which = -F, but gets squared
        self.cy_data = np.array([])
        self.ir_dipole = np.zeros((self.norm_size - 6, 3))
        self.norms_from_mw_hess = np.array([])
        self.eigVals_from_mw_hess = np.array([])
        self.mw_frequencies = np.array([])
        self.norms_from_hess = np.array([])
        self.eigVals_from_hess = np.array([])
        self.hr = np.array([])

    def diag_mw_hessian(self):
        self.eigVals_from_mw_hess, self.norms_from_mw_hess = LA.eigh(self.mw_hessian)
        self.eigVals_from_mw_hess[:6] = 1e-6
        self.mw_frequencies = np.sqrt(self.eigVals_from_mw_hess) * vib_constant

    def diag_hessian(self):
        self.eigVals_from_hess, self.norms_from_mw_hess = LA.eigh(self.mw_hessian)

    def add_file(self, file):
        self.my_file.append(file)

    def import_freq_data(self, file_number=0, hessian=False, mw_hessian=True):
        hessian_parts = np.zeros((int(math.ceil(self.norm_size / 6)), self.norm_size, 6))
        mw_hessian_parts = np.zeros((int(math.ceil(self.norm_size / 6)), self.norm_size, 6))
        record_hes = False
        record_mw_hes = False
        record_norm = False
        record_mass = False
        with open(self.my_file[file_number], mode='r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_f:
                start_freq = 0
                start_fcnst = 0
                start_mass = 0
                start_int = 0
                start_dip = 0
                n = 0
                i = -1
                vector_set = 0
                nmodeCounter = 0  # Keeps track of what normal mode the code is on.
                for line in iter(mmap_f.readline, b''):
                    if b'*' in line and hessian or b'Mass-Weighted Hessian Matrix' in line and hessian:
                        record_hes = False
                        self.hessian = hessian_parts[0]
                        for part in range(1, hessian_parts.shape[0]):
                            self.hessian = np.concatenate((self.hessian, hessian_parts[part]), axis=1)
                    if record_hes:
                        if i >= 0:
                            if i == self.norm_size:
                                n += 1
                                i = 0
                                continue
                            hessian_parts[n, i, :] = np.delete(split(line), 0)
                        i += 1

                    if b'Vectors for Translations and Rotations' in line and mw_hessian:
                        record_mw_hes = False
                        mw_hessian = False
                        self.mw_hessian = mw_hessian_parts[0]
                        for part in range(1, mw_hessian_parts.shape[0]):
                            self.mw_hessian = np.concatenate((self.mw_hessian, mw_hessian_parts[part]), axis=1)
                        zeros_delete = list(range(self.mw_hessian.shape[1] - (self.norm_size % 6), self.mw_hessian.shape[1]))
                        self.mw_hessian = np.delete(self.mw_hessian, zeros_delete, axis=1)

                    if record_mw_hes:
                        if i >= 0:
                            if i == self.norm_size:
                                n += 1
                                i = -1
                                continue
                            splt = split(line)
                            mw_hessian_parts[n, i, 0:splt.size] = splt
                        i += 1

                    if b'TransDip' in line:
                        record_norm = False
                        vector_set += 3
                    if record_norm:
                        if i >= 0:
                            np_strs_norm = split(line)
                            j = vector_set
                            norm_start = 3 * i
                            norm_finish = norm_start + 3
                            for ind in range(3):
                                start = ind * 3
                                finish = start + 3
                                self.norm[norm_start:norm_finish, j] = np_strs_norm[start:finish].T
                                j += 1
                        i += 1
                    if b'Molecular Mass:' in line:
                        record_mass = False
                    if record_mass:
                        if i >= 0:
                            self.masses[i] = split(line)[1]
                        i += 1
                    if b'Hessian of the SCF Energy' in line and hessian:  # initializes recording for hessian matrix
                        i = -1
                        record_hes = True

                    # initializes recording for mass weighted
                    # hessian matrix
                    if b'Mass-Weighted Hessian Matrix:' in line and mw_hessian:
                        i = -2
                        record_mw_hes = True

                    if b'Raman Active:' in line:  # initializes recording for normal modes
                        i = -1
                        record_norm = True

                    if b'Zero point vibrational energy:' in line:  # initializes recording for masses
                        i = -1
                        record_mass = True

                    if b'Frequency:' in line:  # finds the frequency of each normal mode
                        finish_freq = start_freq + 3
                        self.frequency[start_freq:finish_freq] = split(line)
                        start_freq = finish_freq

                    if b'Force Cnst:' in line:  # finds the force constant of each normal mode
                        finish_fcnst = start_fcnst + 3
                        self.ground_force_cnst[start_fcnst:finish_fcnst] = split(line)
                        start_fcnst = finish_fcnst
                    if b'Red. Mass:' in line:  # finds the reduced mass of each normal mode
                        finish_mass = start_mass + 3
                        self.red_mass[start_mass:finish_mass] = split(line)
                        start_mass = finish_mass
                    if b'IR Intens:' in line:  # finds the IR intensity of each normal mode
                        finish_int = start_int + 3
                        self.ir_intensity[start_int:finish_int] = split(line)
                        start_int = finish_int
                    if b'TransDip' in line:  # finds the IR dipole of each normal mode
                        nmodeCounter += 3
                        finish_dip = start_dip + 3
                        if split(line).size == 9:
                            self.ir_dipole[start_dip:finish_dip, :] = np.reshape(split(line), (3, 3))
                        else:
                            print('instance of split(line).size != 9 occured between normal mode', nmodeCounter - 3, 'and', nmodeCounter)
                        start_dip = finish_dip
        return

    def import_force_data(self, file_number=0):
        force_parts = np.zeros((3, 6))
        record_force = False
        with open(self.my_file[file_number], mode='r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_f:
                i = -1
                force_set = 0
                for line in iter(mmap_f.readline, b''):
                    if b'Gradient time' in line:
                        record_force = False
                        i = 0
                        for j in range(force_parts.shape[1]):
                            start = 3 * j + force_set
                            finish = start + 3
                            if start < self.norm_size - 1:
                                self.force[start:finish] = force_parts[:, j]
                    if record_force:
                        if i >= 0:
                            np_strs_force = np.delete(split(line), 0)
                            if i == 3:
                                i = 0
                                for j in range(force_parts.shape[1]):
                                    start = 3 * j + force_set
                                    finish = start + 3
                                    self.force[start:finish] = force_parts[:, j]
                                force_set += 3 * force_parts.shape[1]
                                continue
                            force_parts[i, 0:np_strs_force.size] = np_strs_force
                    i += 1

                    if b'Gradient of the state energy (including CIS Excitation Energy)' in line:
                        i = -1
                        record_force = True
        return

    def import_uvvis_data(self, file_number=0):
        with open(self.my_file[file_number], 'r') as f:
            self.cy_data = np.loadtxt(f, delimiter=',', usecols=range(2, 4))

    def calc_HR(self):
        invcm2hbarfreq = 1.05457182e-34 * (2 * math.pi * speed_of_light * 100) ** 3
        eigvals, norm_modes = LA.eigh(self.mw_hessian)
        eigvals[:6] = 1e-6
        freqs = np.sqrt(eigvals) * vib_constant
        kg_mass = self.masses / avogadro / 1000
        # kg_mass = data.red_mass / avogadro / 1000
        force_J_per_m = self.force * auforce2newton
        mass_force = np.zeros(force_J_per_m.size)
        i = 0
        for f in force_J_per_m:
            mass_force[i] = f / math.sqrt(kg_mass[i // 3])
            i += 1
        force_norm = np.matmul(norm_modes.T, mass_force)
        self.hr = np.multiply(force_norm ** 2, 1 / (2 * (freqs ** 3 * invcm2hbarfreq)))
        self.hr = self.hr[6:]
        return


def abs_spec(vert_trans, frequencies, huang_rhys, temperature, sigma_inhom, time):
    nm2AU = 45.5640
    EVert = nm2AU / vert_trans  # hatrees
    freqAU = frequencies * kToAU
    eigVals = freqAU ** 2
    sigmaAU = sigma_inhom * kToAU
    Kb = 3.168 * 10 ** -6  # Boltzman constant in atomic units
    coth = 1 / np.tanh(np.sqrt(eigVals) / 2 / Kb / temperature)
    osc = np.zeros(time.size, dtype=complex)
    for i, wV in enumerate(freqAU[6:]):
        omegaTime = wV * time
        osc += huang_rhys[i] * (coth[i] * (np.cos(omegaTime) - 1) - 1j * np.sin(omegaTime))
    osc += -1j * EVert * time - (time * sigmaAU) ** 2 / 2
    rspn = np.exp(osc)

    full_spectra = np.fft.fft(rspn)
    num_neg_points = int(time.size / 2)
    spectra = full_spectra[num_neg_points:]
    freq_axis = -1 * (2 * np.pi) / (time[-1] * 4.56 * 10 ** -6) * np.arange(-time.size / 2 + 1, 1, 1)
    return freq_axis, spectra.real, rspn