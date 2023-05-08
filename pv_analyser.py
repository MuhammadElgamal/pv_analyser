'''---------------------- Imported Modules/ packages ------------'''
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
from meteostat import Point, Hourly   # for using geographical data
import pvlib
import pandas as pd
import scipy
from frechetdist import frdist

'''--------------------  Used Constants -------------------------'''

reading_length_threshold = 5        # the length at which a reading is considered effective
extraction_rounds = 10              # number used to extract the IV curve expected
q = 1.60217663e-19                  # Electron Charge
k = 1.380649e-23                    # Boltzmann constant
'''----------------------- Station Data-------------------------'''

station_location = [30.017695, 31.506666, 328]  #latitude longttude elevation
station_zone_time = 2
station_zone_time = 2
station_panel_count = 40
tilt_angle = 26 # taken from an image analysis of the site picture
station_azimuth = 143.1301 # based on analysis of the site areal photo
'''
TEMPERATURE_MODEL_PARAMETERS = {
    'sapm': {
        'open_rack_glass_glass': {'a': -3.47, 'b': -.0594, 'deltaT': 3},
        'close_mount_glass_glass': {'a': -2.98, 'b': -.0471, 'deltaT': 1},
        'open_rack_glass_polymer': {'a': -3.56, 'b': -.0750, 'deltaT': 3},
        'insulated_back_glass_polymer': {'a': -2.81, 'b': -.0455, 'deltaT': 0},
    },
    'pvsyst': {'freestanding': {'u_c': 29.0, 'u_v': 0},
               'insulated': {'u_c': 15.0, 'u_v': 0}}
Several thermal models for the operation of solar cell
}
'''
a, b, deltaT = -3.56, -0.075, 3
'''
SURFACE_ALBEDOS = {'urban': 0.18,
                   'grass': 0.20,
                   'fresh grass': 0.26,
                   'soil': 0.17,
                   'sand': 0.40,
                   'snow': 0.65,
                   'fresh snow': 0.75,
                   'asphalt': 0.12,
                   'concrete': 0.30,
                   'aluminum': 0.85,
                   'copper': 0.74,
                   'fresh steel': 0.35,
                   'dirty steel': 0.08,
                   'sea': 0.06}
'''
albedo = 0.4 # for a sandy location
'''---------------------  Used functions ------------------------'''
#-------------------- For inducing Fictious data ---------------------------------
def awgn(s,SNRdB=35,L=1):
    # author - Mathuranathan Viswanathan (gaussianwaves.com
    # This code is part of the book Digital Modulations using Python
    from numpy import sum, isrealobj, sqrt
    from numpy.random import standard_normal
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
"""
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r
def run_optimizer(func, x0=0 ,ranges=0,  bounds=0, method='brute'):
    # used mainly for parameter extraction
    if method == 'brute':
        # brute force global minimization
        x0, fval, grid, jout = scipy.optimize.brute(func, ranges, full_output=True,
                                  finish=scipy.optimize.fmin)
        loc = np.where(jout == jout.min())
        loc = np.array(list(loc), dtype='int64')
        loc = np.concatenate([np.array([0]), loc.flatten()], axis=0)
        x_target = np.zeros(x0.shape)
        for i in range(len(x0)):
            loc[0] = i
            x_target[i] = grid[tuple(loc)]
        return x_target, jout.min()
    elif method in ['Nelder-Mead', 'SLSQP', 'L-BFGS-B', 'TNC']:
        # the last method is the best
        # can be trapped in a local minimum
        # you need to specifiy x0 and bounds
        res = scipy.optimize.minimize(func, x0, method=method, tol=1e-6,
                                      bounds=bounds)
        return res.x, res.fun
#-------------------- Parameter saving and calling feature -----------------------
def save_variable(object, name):
    with open(name+".pkl", 'wb') as outp:
        pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
def read_variable(name):
    with open(name + ".pkl", 'rb') as inp:
        object = pickle.load(inp)
    return object
#0------------------------- For acquiring location specific data ----------------------

#--------------------------- for the analysis of station Reports-----------------------------------------------
def find_iv (characterisation):
    # Assuming the charcteisation is a single line
    characterisation = characterisation.split()
    id = int(characterisation[0])
    time_stamp = datetime.datetime.strptime(characterisation[1], '%m/%d/%Y.%H:%M:%S')
    voltage = []
    current = []
    for i in range(2,len(characterisation), 2):
        voltage.append(float(characterisation[i]))
    for i in range(3,len(characterisation), 2):
        current.append(float(characterisation[i]))
    current = np.array(current)
    voltage = np.array(voltage)
    return id, time_stamp, voltage, current
def parse_station_report (file_name, full_report=True, atspecific='location',
                          time=[datetime.datetime(2010, 1,1), datetime.datetime(2050, 1,1)],
                          panel_id=1, extract_parameters=False, condition_extraction =True, find_circuit_parameters=False, cell_count=60):
    iv_measurements = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
    for m in range(len(lines)):
        line = lines[m]
        if full_report:
            id, time_stamp, v, i = find_iv(line)
            iv = iv_measurement(current=i, voltage=v, measuring_time=time_stamp,
                                panel_id=id)
            if len(iv.voltage) > reading_length_threshold and all(iv.current >= 0):
                    if extract_parameters:
                        iv.update_temperature()
                        iv.update_irradiance()
                        iv.extract_features(condition_measurment=condition_extraction)
                    iv_measurements.append(iv)
        else:
            line_splitted = line.strip().split()
            if atspecific == 'location':
                # then we append only the specific location required
                if int(line_splitted[0]) == panel_id:
                    id, time_stamp, v, i = find_iv(line)
                    iv = iv_measurement(current=i, voltage=v, measuring_time=time_stamp,
                                        panel_id=id)
                    if len(iv.voltage) > reading_length_threshold and all(iv.current >= 0):
                        if extract_parameters:
                            iv.update_temperature()
                            iv.update_irradiance()
                            iv.extract_features(condition_measurment=condition_extraction)
                        iv_measurements.append(iv)
            elif atspecific == 'time-range':
                # we collect data about specific panel id in a specific timerange
                id, time_stamp, v, i = find_iv(line)
                if type(panel_id) == type(1):
                    if id == panel_id:
                        if time_stamp >= time[0] and time_stamp <= time[1]:
                            iv = iv_measurement(current=i, voltage=v, measuring_time=time_stamp,
                                                panel_id=id)

                            if len(iv.voltage) > reading_length_threshold and all(iv.current >= 0):

                                if extract_parameters:
                                    iv.update_temperature()
                                    iv.update_irradiance()
                                    iv.extract_features(condition_measurment=condition_extraction)
                                iv_measurements.append(iv)
                else:
                    if id in panel_id:
                        if time_stamp >= time[0] and time_stamp <= time[1]:
                            iv = iv_measurement(current=i, voltage=v, measuring_time=time_stamp,
                                                panel_id=id)

                            if len(iv.voltage) > reading_length_threshold and all(iv.current >= 0):

                                if extract_parameters:
                                    iv.update_temperature()
                                    iv.update_irradiance()
                                    iv.extract_features(condition_measurment=condition_extraction)
                                iv_measurements.append(iv)

        # if m == 0:
        #     print("File ", file_name, " is being analysed")
        # elif m == len(lines)-1:
        #     print("\nFile is finished")
        # else:
        #     print('%-4.2f' % ((m + 1) / len(lines) * 100), " % completed")
    if atspecific == 'all-panels-at-instance':
        with open(file_name, 'r') as file:
            text = file.readlines()
        errors = []
        for n in range(len(text)):
            id, time_stamp, v, i = find_iv(text[n])
            err = time_stamp - time
            errors.append(err.__abs__())
        # --------- Finding minimum error
        targetindex = 0
        for i in range(1, len(errors)):
            if errors[i] < errors[targetindex]:
                targetindex = i
        # --------- Choosing range of operation
        # a = targetindex - (station_panel_count + 5) // 2
        # b = targetindex + (station_panel_count + 5) // 2
        a = targetindex - station_panel_count
        b = targetindex + station_panel_count
        if a < 0:
            a = 0
        if b >= len(text):
            b = len(text)
        iv_station = []
        for n in range(a, b):
            id, time_stamp, v, i = find_iv(text[n])
            iv = iv_measurement(current=i, voltage=v, measuring_time=time_stamp,
                                panel_id=id)
            if len(iv.voltage) > reading_length_threshold and all(iv.current >= 0):
                iv_station.append(iv)
        # initiate a zero list containing measurements
        iv_measurements = []
        for n in range(station_panel_count):
            iv_measurements.append(0)
        for characterisation in iv_station:
            iv_measurements[characterisation.panel_id - 1] = characterisation
    if find_circuit_parameters:
        for i in range(len(iv_measurements)):
            iv_measurements[i].equivalent_circuit(cell_count)

    return iv_measurements
def analyse_panel_through_time (source_file, panel_id, year=[2020, 2045], month=[1, 12], day=[1, 30],hour= [0,23], min=[0,59], chosen_parameters= 'mp', cell_count=60):
    # Accepted value for parameters = ['isc', 'voc', 'imp', 'vmp', 'mp', 'fillfac', 'effciency', 'irradiance', 'temprature', 'rs', 'rp', 'idark', 'ideality', 'iph']
    time_range = [datetime.datetime(year[0], month[0], day[0], hour[0], min[0]),
                  datetime.datetime(year[1], month[1], day[1], hour[1], min[1])]
    data = parse_station_report(source_file, full_report=False, atspecific='time-range', panel_id=panel_id,
                                   time=time_range, extract_parameters=True, find_circuit_parameters=True, cell_count=cell_count)
    time = []
    for i in range(len(data)):
        time.append(data[i].measuring_time.strftime('%m-%d %H:%M'))
    all_parameter_values = []
    for p in chosen_parameters:
        parameter = []
        for i in range(len(data)):
            code = "parameter.append(data[i]." + p + ")"
            exec(code)
        all_parameter_values.append(parameter)
    df = pd.DataFrame()
    df['time'] = time
    for p in range(len(chosen_parameters)):
        df[chosen_parameters[p]] = all_parameter_values[p]
    df = df.set_index('time')
    return df
def analyse_station_across_space (source_file,time_of_analysis = datetime.datetime(2022, 3, 10, 16, 20),
                                    panels_included = [1, 2],
                                    chosen_parameters = ['mp'],
                                    time_resolution = 50000):
    # ------------------- Collecting search data -------------------------------------------------
    time_resolution = datetime.timedelta(seconds=time_resolution)
    trials = 0
    trial_max = 10
    while trials < trial_max:
        data = []
        time_range = [time_of_analysis - time_resolution / 2, time_of_analysis + time_resolution / 2]
        # for panel in panels_included:
        #     a = parse_station_report(source_file, full_report=False,
        #                                 atspecific='time-range', panel_id=panel, time=time_range,
        #                                 extract_parameters=True)
        #     # -------------- To make sure that each panel has at least two readings
        #     if len(a) >= 2:
        #         data.append(a)  # so each item in the least is the several readings for the same panel
        #     else:
        #         break
        #------------------------
        datan = parse_station_report(source_file, full_report=False,
                                        atspecific='time-range', panel_id=panels_included, time=time_range,
                                        extract_parameters=True)
        for i in range(len(panels_included)):
            a = []
            for j in range(len(datan)):
                if datan[j].panel_id == panels_included[i]:
                    a.append(datan[j])
            data.append(a)
        #-----------------------
        total_len = 0
        for d in data:
            total_len += len(d)
        trials += 1
        if total_len < len(panels_included) * 2 and len(data) != len(panels_included):
            time_resolution *= 2
        else:
            break
    if trials >= trial_max:
        print("Data is insufficient for judgement")
        print("Resolution Time is ", time_resolution, "hours")
        import sys
        sys.exit(1)
    else:
        print("Time Resolution suffcient for each panel to be charcterised twice is ", time_resolution, "hour")
        print("Number of search items ", len(data))

    # ----------------------------- Interpolate readings -----------------------------------------
    # we need first to find closest matching readings for each panel and then do interpolation
    # for each panel_item we need to define the lower and upper then do interpolation
    panel_measurements = []
    for p in range(len(panels_included)):
        # find the nearest below in time and the maximum up in time
        time_diff = np.zeros((len(data[p]), 1))
        for d in range(len(data[p])):
            time_diff[d] = np.abs((data[p][d].measuring_time - time_of_analysis).seconds)
        time_diff = time_diff.transpose()[0]
        print(time_diff)
        order = time_diff.argsort()
        print(order)
        panel_measurements.append(data[p][order[0]].time_interpolate(data[p][order[1]], time_of_analysis))
    # ------------------- Extracting Required Parameters as a data frame --------------------------------------------
    all_parameter_values = []
    for p in chosen_parameters:
        parameter = []
        for i in range(len(panel_measurements)):
            code = "parameter.append(panel_measurements[i]." + p + ")"
            exec(code)
        all_parameter_values.append(parameter)

    df = pd.DataFrame()
    df['Location'] = panels_included
    for p in range(len(chosen_parameters)):
        df[chosen_parameters[p]] = all_parameter_values[p]
    df = df.set_index('Location')
    return df
'''--------------------------- Defined Classes -----------------'''
class panel_diode:
    def __init__(self,reverse_current=1e-3, temperature=25, ideality=1.6):
        # This data are for a specific common rectifier diode used for solar panels that can be found here
        # https://www.diodes.com/search/?q=bypass+diode+solar+panels&t=keyword&action_results=Go
        self.reverse_current = reverse_current
        self.temperature = temperature
        self.ideality = ideality
    def find_voltage(self,current=np.array([1])):
        s = self.ideality * k * (self.temperature + 273.15) / q * np.log(current / self.reverse_current + 1)
        self.voltage = s
        self.current = current
class iv_measurement:
    def __init__(self, measuring_time = 'N/A', acquiry_time = 0.0, temprature = 25, irradiance = 1000, current = np.zeros(1000), voltage= np.zeros(1000), panel_id = 1):
        # measuring time signifies the time when measurement was taken from site
        # acquiry time signifies the time wher simulation was done
        self.measuring_time = measuring_time
        self.acquiry_time = acquiry_time
        self.temprature = temprature
        self.irradiance = irradiance
        # the voltage property includes measurement whether conditioned or not
        # the measured voltage property includes the original measurement whatever its source
        # the simulated property is only filled when there is a simulation to check the performance
        self.current = current
        self.voltage = voltage
        self.measured_voltage = voltage
        self.measured_current = current
        self.simulated_voltage = voltage
        self.simulated_current = current
        self.panel_id = panel_id
        self.vmin = self.voltage.min()
        self.vmax = self.voltage.max()
        self.isc = []
        self.voc = []
        self.mp = []
        self.imp = []
        self.vmp = []
        self.fillfac = []
        self.effciency = []
        self.rs = []
        self.rp = []
        self.idark = []
        self.iph = []
        self.ideality = []
        self.equivalent_circuit_found = True
        #--------------- Updated mainly during a simulation process --------------------------
        self.state_matrix = [] # includes circuit parameters for state_matrix = [rs, rp, iph, idark, ideality, temp]
    def extract_features(self, condition_measurment=False, conditioning_threshold=0.05, point_count = 1000, fitting_order = 10, find_current_modes=False, diode_count=3):
        # point_count must be optimally 20 * fitting order
        if condition_measurment:
            # ----------------- Taking voltage and current and sorting them ---------------------------------------
            v = self.voltage
            i = self.current
            order = v.argsort()
            v = v[order]
            i = i[order]
            # ----------------- Linear Interpolation -------------------------------------------------------------
            v_new = np.linspace(v.min(), v.max(), point_count)
            i_new = np.interp(v_new, v, i)
            v, i = v_new, i_new
            # --------------------------- Differntiation and saturation removal ----------------------------------
            try:
                deriv = abs(np.diff(i) / np.diff(v))
                max_loc = np.array(np.where(deriv == deriv.max()))
                max_loc = max_loc.reshape((max_loc.size, 1))
                a = np.array(np.where(deriv < conditioning_threshold))
                a = a[a > max_loc[-1]]
                v = v[0:a[0]]
                i = i[0:a[0]]
            except:
                print("iv curve doesnot have current saturation anomaly")

            # ---------------------  adding initial values -----------------------------------------------------
            v_init = np.linspace(0, v.min(), 100)
            i_init = np.average(i_new[0:50]) * np.ones(v_init.shape)
            v = np.concatenate([v_init, v])
            i = np.concatenate([i_init, i])
            # sample data unifromly
            v_new = np.linspace(v.min(), v.max(), point_count)
            i_new = np.interp(v_new, v, i)
            v, i = v_new, i_new
            # ------------------------ Extrapolating to find Voc ----------------------------------------------
            while fitting_order > 1:
                for voc_extrapolation_factor in range(100):
                    last_element = -int(voc_extrapolation_factor / 100 * point_count)
                    fitting = np.poly1d(np.polyfit(v[last_element:-1], i[last_element:-1], fitting_order))
                    vq = np.linspace(v[last_element], v[-1] + 30, point_count)
                    iq = fitting(vq)
                    if iq.max() == iq[0]:
                        break
                if not all(np.diff(iq) < 0):
                    if fitting_order != 1:
                        fitting_order -= 1
                else:
                    break

            v = v[0: last_element]
            i = i[0: last_element]
            v = np.concatenate([v, vq])
            i = np.concatenate([i, iq])
            v = v[i >= 0]
            i = i[i >= 0]
            # sample data unifromly
            v_new = np.linspace(v.min(), v.max(), point_count)
            i_new = np.interp(v_new, v, i)
            v, i = v_new, i_new
            self.voltage = v
            self.current = i
            if find_current_modes:
                # The method is effcient when the SNR of the measurement is as low as 20dB
                peak_threshold = 0.2  # peak threshold considers which value is considered an actual peak in the histogram as a ratio of the maximum peak
                hist, bin_edges = np.histogram(i, density=True,
                                               bins=1000)  # the bin count is chosen to gaurantee that the histogram truely captures actual distribution of parameters
                bin_center = (bin_edges[0:-1] + bin_edges[1:]) / 2
                N = 5  # low number of points after the end means that the interpolating polynomial has lower oscillations which affects peak detection
                # very low N can misdetect the biggest peak and very high N makes interpolating polynomial oscilatroy
                hist = np.concatenate([hist, np.zeros((N,))], axis=0)
                delta = bin_center[1] - bin_center[0]
                max_bin = N * delta + bin_center.max()
                bin_center = np.concatenate([bin_center, np.linspace(bin_center.max(), max_bin, N)], axis=0)
                # fitting is preferrable over other filtering methods like moving averages as these methods slightly shift the peaks
                fitting = np.poly1d(np.polyfit(bin_center, hist,
                                               100))  # higher order polynomial is useful when peaks are very close to each other (when differnt in current levels is 0.5 A for instance)
                hist_new = fitting(bin_center)
                hist_new /= hist_new.max()
                peaks = scipy.signal.find_peaks(hist_new)
                order = np.flip(hist_new[peaks[0]].argsort())
                target_peaks = peaks[0][order[0:diode_count]]
                values = hist_new[target_peaks]
                target_peaks = target_peaks[values > peak_threshold]
                modes = bin_center[target_peaks]
                self.current_modes = modes
        self.power = self.voltage * self.current
        self.isc = self.current.max()
        self.voc = self.voltage[self.current >= 0].max()
        self.mp = self.power.max()
        self.imp = float(self.current[self.power == self.mp])
        self.vmp = float(self.voltage[self.power == self.mp])
        self.fillfac = self.mp / (self.voc * self.isc)
        if self.temprature != 25:
            self.update_temperature()
        if self.irradiance != 1000:
            self.update_irradiance()
        self.effciency = self.mp / self.irradiance # unit is /m^2
    def equivalent_circuit(self, N=60, method='modified_laudani'):
        # N means cells in series
        # the method is based on the way described in https://www.mdpi.com/1996-1073/12/22/4271
        # N is number of cells in series
        # point_count is number of points within circuit charcterisation
        try:
            if method == 'stornelli':
                measurement = self
                measurement.extract_features(condition_measurment=True)
                from math import exp, log
                # Code implemented from matlab
                T = self.temprature + 273.15  # Temperature in Kelvin
                ## Datasheet table STC value of KC200GT panel
                Isc = measurement.isc  # Short circuit current
                Voc = measurement.voc  # Open circuit voltage
                Imp = measurement.imp  # Maximum power current
                Vmp = measurement.vmp  # Maximum power voltage
                Pmax = Vmp * Imp  # Maximum power point
                A = 1
                vt = (k * A * T * N) / q
                Rs = (Voc / Imp) - (Vmp / Imp) + ((vt / Imp) * log((vt) / (vt + Vmp)))
                I0 = Isc / (exp(Voc / vt) - exp(Rs * Isc / vt))
                Ipv = I0 * ((exp(Voc / vt)) - 1)
                # ------------- first step
                iter = 1000
                it = 0
                tol = 0.01
                A1 = A
                VmpC = (vt * (log((Ipv + I0 - Imp) / I0))) - (Rs * Imp)
                e1 = VmpC - Vmp
                Rs1 = Rs

                while (it < iter and e1 > tol):
                    if VmpC < Vmp:
                        A1 = A1 - 0.01
                    else:
                        A1 = A1 + 0.01
                    vt1 = (k * A1 * T * N) / q
                    I01 = Isc / (exp(Voc / vt1) - exp(Rs1 * Isc / vt1))
                    Ipv1 = I01 * ((exp(Voc / vt1)) - 1)
                    VmpC = (vt1 * (log((Ipv1 + I01 - Imp) / I01))) - (Rs1 * Imp)
                    e1 = (VmpC - Vmp)
                    it = it + 1
                vt1 = (k * A1 * T * N) / q
                Rs1 = (Voc / Imp) - (VmpC / Imp) + ((vt1 / Imp) * log((vt1) / (vt1 + VmpC)))
                # ---------------------- Second step ---------------------------------
                tolI = 0.001
                iter = 10000
                itI = 0
                I01 = Isc / (exp(Voc / vt1) - exp(Rs1 * Isc / vt1))
                Ipv1 = I01 * ((exp(Voc / vt1)) - 1)
                Rp = ((- Vmp) * (Vmp + (Rs1 * Imp))) / (
                            Pmax - (Vmp * Ipv1) + (Vmp * I01 * (exp(((Vmp + (Rs1 * Imp)) / vt1) - 1))))
                # =------ calculate I0 with new Rp value -----------------------------------
                I02 = (Isc * (1 + Rs1 / Rp) - Voc / Rp) / (exp(Voc / vt1) - exp(Rs1 * Isc / vt1))
                Ipv2 = I02 * ((exp(Voc / vt1)) - 1) + Voc / Rp
                ImpC = Pmax / VmpC
                Err = abs(Imp - ImpC)
                Rpnew = Rp
                err = 1e10
                while err > tolI and itI < iter:
                    if ImpC < Imp:
                        Rpnew = Rp + 0.1 * itI
                    elif ImpC >= Imp:
                        Rpnew = Rp - 0.1 * itI

                    # ----Calculate I0 with Rpnew
                    I02 = (Isc * (1 + Rs1 / Rpnew) - Voc / Rpnew) / (exp(Voc / vt1) - exp(Rs1 * Isc / vt1))
                    Ipv2 = I02 * ((exp(Voc / vt1)) - 1) + Voc / Rpnew

                    def func(ImpC):
                        return Ipv2 - (I02 * (exp((Vmp + (Rs1 * ImpC)) / vt1) - 1)) - ImpC - (Vmp + Rs1 * ImpC) / Rpnew

                    from scipy.optimize import root_scalar
                    # ImpC = root_scalar(func, method='toms748', bracket=[0.5 * Imp, 1.5 * Imp])
                    # to avoid bracketing errors
                    ImpC = root_scalar(func, method='toms748', bracket=[0, 5 * Imp])

                    ImpC = ImpC.root
                    itI = itI + 1
                    err = abs(Imp - ImpC)

                # -------------------- parsing output ----------------------------------
                rs = Rs1
                rp = Rpnew
                ideality = A1
                idark = I02
                iph = Ipv2

                # -------------- Building the circuit model ----------------------------
                self.rs = rs
                self.rp = rp
                self.idark = idark
                self.iph = iph
                self.ideality = ideality
            elif method == 'modified_laudani':
                '''
                https://www.sciencedirect.com/science/article/pii/S1364032118300984?casa_token=h29icoynxasAAAAA:BzsRQ-caCSVfpyw4OFYyqDBoyKTQjr1_qFnv2u5V6_xg3x_gJhL8kZlA1dhUIk-_EdelatxzDA 
                '''
                '''
                Used model from the review paper
                https://www.sciencedirect.com/science/article/pii/S0038092X13005203?casa_token=9X7ZpMGPvpoAAAAA:iFePpgut24wwrsAgXvxhOQJ1COS26hEfqGc17MmtW0yMgNLIqlQbI5GGGOoDinZZ5iIFYKuuVw
                https://www.sciencedirect.com/science/article/pii/S0038092X14000929?casa_token=GWh_JYk-10QAAAAA:wCwYCOrbS9MbBYDRump2p7Uv-MrhL-MYckqxD3xXFOM4aNCN44uHxSEM3SQVkFLE7JdUfX2Bgg
                '''
                # self.update_temperature()
                self.extract_features(condition_measurment=True, point_count=200)
                T = self.temprature + 273.15
                vt = k * T / q
                v = self.voltage
                i = self.current
                isc = self.isc
                voc = self.voc
                imp = self.imp
                vmp = self.vmp
                # ----------- Te value theta is the ideality followed by the series resistance
                theta_0 = np.zeros((2,))
                theta_0[0] = 1
                theta_0[1] = N * theta_0[0] * vt / imp * (1 + scipy.special.lambertw(
                    -np.exp((voc - 2 * vmp - N * theta_0[0] * vt) / (N * theta_0[0] * vt)))) + vmp / imp

                # theta[1] is the upper bound of theta[1], theta[0] is the initial guess of theta[0]
                def calculate_parameters(theta, method='RF2'):
                    n = theta[0]
                    rs = theta[1]
                    exp_oc = np.exp(voc / (N * n * vt))
                    exp_mpp = np.exp((vmp + rs * imp) / (N * n * vt))
                    exp_sc = np.exp((rs * isc) / (N * n * vt))
                    # ---------- Going to the reduced form -------------------
                    if method == 'RF2':
                        a1 = vmp + rs * imp - voc
                        a2 = voc - rs * isc
                        a3 = rs * isc - rs * imp - vmp
                        gsh = (exp_oc * (imp - isc) + exp_mpp * isc - exp_sc * imp) / (
                                    a1 * exp_sc + a2 * exp_mpp + a3 * exp_oc)
                        rp = 1 / gsh
                        idark = (voc * (isc - imp) - vmp * isc) / (a1 * exp_sc + a2 * exp_mpp + a3 * exp_oc)
                        iph = (isc * voc * (exp_mpp - 1) + isc * vmp * (1 - exp_oc) + imp * voc * (1 - exp_sc)) / (
                                    a1 * exp_sc + a2 * exp_mpp + a3 * exp_oc)
                    elif method == 'RF1':
                        denominator = (rs * imp - vmp) * (
                                    exp_mpp * (imp * rs + vmp - voc - N * n * vt) + exp_oc * N * n * vt)
                        gsh = (imp * ((vmp + N * n * vt - imp * rs) * exp_mpp - N * n * vt * exp_oc)) / denominator
                        rp = 1 / gsh
                        idark = imp * (voc - 2 * vmp) * N * n * vt / denominator
                        iph = imp * (voc * (vmp - imp * rs + N * n * vt) * exp_mpp + N * n * vt * (
                                    2 * vmp * (1 - exp_oc) - voc)) / denominator
                    parameters = np.array([iph, idark, rs, rp, n])  # parameters are in this specific order
                    parameters[parameters < 0] *= -1
                    return list(parameters)

                    x = 1
                def cost_function(parameters):
                    iph = parameters[0]
                    idark = parameters[1]
                    rs = parameters[2]
                    rp = parameters[3]
                    ideality = parameters[4]

                    i_produced = np.zeros(i.shape)
                    # ------- Do simulation -----------------
                    for j in range(len(v)):
                        def sdm(c):
                            # c stands for current
                            return iph - idark * (np.exp((v[j] + c * rs) / (N * ideality * vt)) - 1) - (
                                        v[j] + c * rs) / rp - c

                        c = scipy.optimize.fsolve(sdm, (2))
                        i_produced[j] = c
                    minmax_err = np.abs((i_produced - i) * v).max()
                    return minmax_err, i_produced
                def example_function(theta):
                    # the RS value is given as a logarithm
                    theta[1] = 10 ** (theta[1])
                    parameters = calculate_parameters(theta)
                    err = cost_function(parameters)
                    # In order of maximum error per measurement minmax<frechet<mse
                    return err[0]
                x0 = [1.01, -6]
                bounds = [(0.5, 2.5), (-6, np.log10(theta_0[1]))]


                theta_min1, fval1 = run_optimizer(example_function, x0=x0, bounds=bounds, method='TNC')
                # if fval1 > 0.02 * isc:
                #     ranges = [slice(1.5, 1.8, 0.02), slice(-6, np.log10(theta_0[1]), 0.1)]
                #     theta_min0, fval0 = run_optimizer(example_function, x0=x0, bounds=bounds, method='brute',
                #                                       ranges=ranges)
                #     if fval1 < fval0:
                #         theta_min = theta_min1
                #         self.fitting_error = fval1
                #     else:
                #         theta_min = theta_min0
                #         self.fitting_error = fval0
                # else:
                #     theta_min = theta_min1
                theta_min = theta_min1
                theta_min[1] = 10 ** theta_min[1]
                # --------- Drawing resulting curve ----------------------------------
                parameters = calculate_parameters(theta_min)
                minmax_err, i_produced = cost_function(parameters)
                self.rs = parameters[2]
                self.rp = parameters[3]
                self.idark = parameters[1]
                self.iph = parameters[0]
                self.ideality = parameters[4]
                self.simulated_current = i_produced
                self.simulated_voltage = v
        except:
            self.equivalent_circuit_found = False
    def find_sun_time(self):
        day = datetime.datetime(year=self.measuring_time.year,
                                month=self.measuring_time.month,
                                day=self.measuring_time.day)
        day_pd = pd.DatetimeIndex([datetime.datetime.strftime(day, "%Y-%m-%d %H:%M:%S")], tz=2)
        sun_data = pvlib.solarposition.sun_rise_set_transit_spa(day_pd, station_location[0], station_location[1])
        sunset = sun_data['sunset'].values[0]
        sunset = (sunset - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        sunset = datetime.datetime.fromtimestamp(sunset)
        sunrise = sun_data['sunrise'].values[0]
        sunrise = (sunrise - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        sunrise = datetime.datetime.fromtimestamp(sunrise)
        self.sunrise = sunrise
        self.sunset = sunset
    def display_features(self):
        print('Displayed Features: ')
        print('Isc  = ', self.isc, ' A')
        print('Voc  = ', self.voc, ' V')
        print('Vmin = ', self.vmin, ' V')
        print('Maximum Power = ', self.mp, ' W')
        print('Imp = ', self.imp, ' A')
        print('Vmp = ', self.vmp, ' V')
        print('Fill Factor = ', self.fillfac)
        print('Effciency = ', self.effciency)
        print('_______________________________')
    def plot(self, several_plots=False, duration=1.3):
        # usual or poly-fitted
        fig, ax = plt.subplots()
        ax.plot(self.voltage, self.current)

        ax.set(xlabel='Voltage (V)', ylabel='Current (A)',
               title='Measured at ' + str(self.measuring_time) + ' T= '+ str(self.temprature)+ ' Celsius '+ ' G='+ "%4.0f"%(self.irradiance)+ ' mSUN')
        #plt.ylim(0, 1.1 * self.isc)
        plt.ylim(0, 15)
        plt.xlim(0, self.voltage.max()*1.5)
        ax.grid()
        if several_plots:
            plt.show(block=False)
            plt.pause(duration)
            plt.close()
        else:
            plt.show()
        return fig, ax
    def update_temperature(self):
        self.update_irradiance()
        def find_temperature(time):
            # Based on the library described here
            # https://dev.meteostat.net/python/normals.html#example
            # time is given as in Egypt time so it needs to be differenced
            time1 = datetime.datetime(time.year, time.month, time.day, time.hour) - datetime.timedelta(
                hours=station_zone_time)
            loc = Point(station_location[0], station_location[1], station_location[2])
            data = Hourly(loc, time1, time1).fetch()
            air_temp = data.at[str(time1), 'temp'] # contains the air temperature for the cell
            wind_speed = data.at[str(time1), 'wspd'] /3.6  # to be in m/s not Km/hr
            c = pvlib.temperature.sapm_cell(self.irradiance, air_temp, wind_speed, a, b, deltaT)
            return c
        self.temprature = find_temperature(self.measuring_time)
    def update_irradiance(self):
        def find_irradiance(time):
            # based on
            # https://pysolar.readthedocs.io/en/latest/#references
            '''
            # Depending on the PySolar Library which is simple and inaccurate
            time_zone_diff = station_zone_time
            latitude_deg = station_location[0]  # positive in the northern hemisphere
            longitude_deg = station_location[1]  # negative reckoning west from prime meridian in Greenwich, England
            time1 = time - datetime.timedelta(hours=time_zone_diff)
            date = datetime.datetime(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second, 0,
                                     tzinfo=datetime.timezone.utc)
            altitude_deg = pysolar.solar.get_altitude(latitude_deg, longitude_deg, date)
            a = pysolar.radiation.get_radiation_direct(date, altitude_deg)
            '''
            Location = pvlib.location.Location(station_location[0], station_location[1], tz=station_zone_time,
                                               altitude=station_location[2])
            time_pd = pd.DatetimeIndex([datetime.datetime.strftime(time, "%Y-%m-%d %H:%M:%S")])
            loc = Point(station_location[0], station_location[1], station_location[2])
            time1 = datetime.datetime(time.year, time.month, time.day, time.hour) - datetime.timedelta(
                hours=station_zone_time)
            data = Hourly(loc, time1, time1).fetch()
            temperature = data.at[str(time1), 'temp']
            pressure = data.at[str(time1), 'pres'] * 100
            solar_position = Location.get_solarposition(time_pd, pressure=pressure, temperature=temperature)
            # https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.location.Location.get_clearsky.html#pvlib.location.Location.get_clearsky
            irradiance = Location.get_clearsky(time_pd, model='ineichen', solar_position=solar_position)
            # ‘ineichen’, ‘haurwitz’, ‘simplified_solis’
            ghi = irradiance.ghi.values[0]
            # we can calculate the dni from pysolar as it has more accurate model
            # time_zone_diff = station_zone_time
            # latitude_deg = station_location[0]  # positive in the northern hemisphere
            # longitude_deg = station_location[1]  # negative reckoning west from prime meridian in Greenwich, England
            # time1 = time - datetime.timedelta(hours=time_zone_diff)
            # date = datetime.datetime(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second, 0,
            #                          tzinfo=datetime.timezone.utc)
            # altitude_deg = pysolar.solar.get_altitude(latitude_deg, longitude_deg, date)
            # a = pysolar.radiation.get_radiation_direct(date, altitude_deg)
            # ------------------------------------------
            dni = irradiance.dni.values[0]
            # dni = a
            dhi = irradiance.dhi.values[0]
            air_mass = Location.get_airmass(times=time_pd, solar_position=solar_position)
            dni_extra = pvlib.irradiance.get_extra_radiation(time,
                                                             method='spencer', epoch_year=time.year)
            total_irrad = pvlib.irradiance.get_total_irradiance(tilt_angle, station_azimuth,
                                                                solar_position['apparent_zenith'].values[0],
                                                                solar_position['azimuth'].values[0],
                                                                dni, ghi, dhi,
                                                                airmass=air_mass['airmass_relative'].values[0],
                                                                albedo=albedo,
                                                                model='perez', model_perez='allsitescomposite1990',
                                                                dni_extra=dni_extra)
            return total_irrad['poa_global']
        self.irradiance = find_irradiance(self.measuring_time - datetime.timedelta(hours=station_zone_time))
    def time_interpolate(r1, r2, t):
        point_count = 1000
        # ------------------- Condition both measurements at the begining ----------------------
        r1.extract_features(condition_measurment=True)
        r2.extract_features(condition_measurment=True)
        v1 = r1.voltage
        i1 = r1.current
        v2 = r2.voltage
        i2 = r2.current
        vmax1 = v1.max()
        vmax2 = v2.max()
        vmax = max(vmax1, vmax2)
        # ---------- Make both signals same length but with packing zeros ---------------------
        v = np.linspace(0, vmax, point_count)
        # ----- Finding both maximums in the single array v
        loc1 = np.array(np.where(v < vmax1)).max()
        loc2 = np.array(np.where(v < vmax2)).max()
        ires1 = np.zeros(v.shape)
        ires2 = np.zeros(v.shape)
        # print(np.interp(v[0:loc1+1], v1, i1))
        ires1[0:loc1 + 1] = np.interp(v[0:loc1 + 1], v1, i1)
        ires2[0:loc2 + 1] = np.interp(v[0:loc2 + 1], v2, i2)
        #------------------- Resultant current --------------------------------------------------
        y1 = ires1
        y2 = ires2
        x1 = r1.measuring_time
        x2 = r2.measuring_time
        x = t
        y = y1 + ((x - x1)/(x2 - x1)) * (y2 - y1)
        result = iv_measurement(measuring_time=t, current=y, voltage=v, panel_id=r1.panel_id)
        result.update_temperature()
        result.update_irradiance()
        result.extract_features(condition_measurment=True)
        return result
    def find_dissimilarity(self, measurement):
        v_s = self.voltage
        i_s = self.current
        v_t = measurement.voltage
        i_t = measurement.current
        i_t = i_t[v_t <= v_s.max()]
        v_t = v_t[v_t <= v_s.max()]
        vs_new = np.linspace(0,v_s.max())
        is_new = np.interp(vs_new, v_s, i_s)
        it_new = np.interp(vs_new, v_t, i_t)
        vs_new /= vs_new.max()
        it_new /= is_new.max()
        is_new /= is_new.max()
        P = np.vstack([vs_new, is_new])
        Q = np.vstack([vs_new, it_new])
        return frdist(P,Q)
    def save_as_mat(self, name):
        mdic = {"measured_voltage":self.measured_voltage,
            "simulated_voltage": self.simulated_voltage,
            "measuring_time": [
                self.measuring_time.year,
                self.measuring_time.month,
                self.measuring_time.day,
                self.measuring_time.hour,
                self.measuring_time.minute,
                self.measuring_time.second
            ] ,
            "temperature": self.temprature,
            "irradiance": self.irradiance,
            "conditioned_current": self.current,
            "conditioned_voltage": self.voltage,
            "measured_current": self.measured_current,
            "measured_voltage": self.measured_voltage,
            "simulated_voltage":self.simulated_voltage,
            "simulated_current":self.simulated_current,
            "panel_id": self.panel_id,
            "vmin": self.vmin,
            "vmax":self.vmax,
            "isc":self.isc,
            "voc":self.voc,
            "mp":self.mp,
            "imp":self.imp,
            "fillfac":self.fillfac,
            "efficiency":self.effciency,
            "rs": self.rs,
            "rp": self.rp,
            "idark": self.idark,
            "iph": self.iph,
            "ideality":self.ideality}
        scipy.io.savemat(name, mdic)
    def read_from_mat(self, file_name):
        mat = scipy.io.loadmat(file_name)
        self.measured_voltage = mat['measured_voltage'].flatten()
        self.simulated_voltage = mat['simulated_voltage'].flatten()
        t = mat['measuring_time'].flatten()
        self.measuring_time = datetime.datetime(
            year=t[0],
            month=t[1],
            day=t[2],
            hour=t[3],
            minute=t[4],
            second=t[5]
        )
        self.temprature = mat['temperature'].flatten()[0]
        self.irradiance = mat['irradiance'].flatten()[0]
        self.current =  mat['conditioned_current'].flatten()
        self.voltage = mat['conditioned_voltage'].flatten()
        self.measured_current = mat['measured_current'].flatten()
        self.measured_voltage = mat['measured_voltage'].flatten()
        self.simulated_voltage = mat['simulated_voltage'].flatten()
        self.simulated_current = mat['simulated_current'].flatten()
        self.panel_id = mat['panel_id'].flatten()[0]
        self.vmin = mat['vmin'].flatten()[0]
        self.vmax = mat['vmax'].flatten()[0]
        self.isc =mat['isc'].flatten()[0]
        self.voc = mat['voc'].flatten()[0]
        self.mp = mat['mp'].flatten()[0]
        self.imp = mat['imp'].flatten()[0]
        self.fillfac = mat['fillfac'].flatten()[0]
        self.effciency = mat['efficiency'].flatten()[0]
        if mat['rs'].size > 0:
            self.rs = mat['rs'].flatten()[0]
            self.rp = mat['rp'].flatten()[0]
            self.idark = mat['idark'].flatten()[0]
            self.iph = mat['iph'].flatten()[0]
            self.ideality = mat['ideality'].flatten()[0]
class cell:
    def __init__(self,
                 rp=np.array([1e6]),
                 rs=np.array([0.1e-3]),
                 iph=np.array([9]),
                 temp=np.array([25]),
                 idark=np.array([0.1e-9]),
                 ideality=np.array([1.1]),
                 area=0.15**2):
        self.rp = rp                                       # Parallel Resistance in One Diode Model
        self.rs = rs                                       # Series Resistance in One Diode Model
        self.iph = iph                                     # Incident Irradiance in W/m^2
        self.temp = temp                                   # Temprature in celesius
        self.idark = idark                                 # Reverse Saturation Current
        self.ideality = ideality                           # Ideality factor for diode contained in cell model
        self.area = area                                   # Area of single cell in units of m^2
    def characterize(self, current=np.array([1]), start_guess=2):
        # Ns represents cells in series in case this class was used for panel simulation
        self.current = current
        def solve_cell(I, T=25, Iph=9, Idark=0.1e-9, rs=10e-3, rp=1e6, ideality=1.1, start_guess=2):
            T = T + 273.15
            s = q / (ideality * k * T)
            V = np.zeros(I.shape)
            for i in range(len(I)):
                def SDM(p):
                    return (Iph - Idark * (np.exp(s * (p + I[i] * rs)) - 1) - (p + I[i] * rs) / rp - I[i])

                v = scipy.optimize.fsolve(SDM, (start_guess))
                V[i] = v
            return V
        self.voltage = solve_cell(current,
                                  T = self.temp,
                                  Iph= self.iph,
                                  Idark=self.idark,
                                  rs=self.rs,
                                  rp=self.rp,
                                  ideality=self.ideality,
                                  start_guess=start_guess
        )
        V = self.voltage
        V = V[current <= self.iph * 1.01]
        current = current[current <= self.iph * 1.01]
        current = current[V>=0]
        V = V[V>=0]
        order = V.argsort()
        V = V[order]
        current = current[order]
        v_new = np.linspace(0, V.max(), 1000)
        i_new = np.interp(v_new, V, current)
        self.measurement = iv_measurement(
            measuring_time=datetime.datetime.now(),
            current=i_new,
            voltage=v_new
        )
        # self.measurement.extract_features()
class cell_string(cell):
    def characterize(self, current=np.array([1]), start_guess=2):
        self.current = current
        V = np.zeros(self.current.shape)
        for i in range(len(self.iph)):
            subcell = cell(
                rp= self.rp[i],
                rs=self.rs[i],
                iph=self.iph[i],
                temp=self.temp[i],
                idark=self.idark[i],
                ideality=self.ideality[i]
            )
            subcell.characterize(current=self.current, start_guess=start_guess)
            V += subcell.voltage
        V[self.current > self.iph.min() * 1.05] = 0  # to clean out any spurious noise in the solution
        self.voltage = V
        V = V[current <= self.iph.min() * 1.01]
        current = current[current <= self.iph.min() * 1.01]
        current = current[V >= 0]
        V = V[V >= 0]
        order = V.argsort()
        V = V[order]
        current = current[order]
        v_new = np.linspace(0, V.max(), 1000)
        i_new = np.interp(v_new, V, current)
        self.measurement = iv_measurement(
            measuring_time=datetime.datetime.now(),
            current=i_new,
            voltage=v_new
        )
        self.measurement.extract_features()
class panel(cell_string):
    # Default panel data is found on this site
    def __init__(self):
        """-------------------------- Manufacturing Model Parameters ------------------------------------------------"""
        self.model_name = 'STP-275-20_Wfw/Suntech'
        # Panel Geometry
        self.rows = 10
        self.columns = 6
        self.cell_string_count = 3
        self.area = 1.65 * 0.992  # in m^2
        # Panel Electrical Charcterisitics
        self.isc = 9.27
        self.voc = 38.1
        self.vmp = 31.2
        self.imp = 8.82
        self.mp = 275
        self.efficiency = 16.8  # as a percentage
        # Thermal ratings
        self.power_thermal_coeffcient = -0.41/100*200.6  # percentage converted to W/C
        self.voc_thermal_coeffcient = -0.33/100*34.8# V/C
        self.isc_thermal_coeffcient = 0.067/100*7.5#A/C converted from datasheet percent value at NOCT conditions
        self.isc_nonlinearity_factor = 1.0206 # the nonlinearity factor is ln(Isc1 / Isc2) / ln(G1/G2) at maximum and minimum irradiance measurement
        self.voc_irradiance_correction_factors =  [ -0.011901070649365119, 1.5320837094311133] # used to iterpolate the dependence of voc on irradiance
        # Panel Test Conditions
        self.stc_irrad = 1000
        self.stc_temp = 25
        self.stc_spectrum = 'AM1.5'

        # used diode in the simulation
        self.diode = panel_diode()
        # Actual measurement Vs. datasheet measurement
        self.ideal_measurement = iv_measurement()
        self.measurements = []
    def load_measurement(self, source_type='csv', source='None', type= 'STC', find_equivalent_circuit=False):
        # type can be simulated, taken or STC
        if source_type == 'measurement' and source =='None':
            raise Exception('Enter a true measurement not a text value')
        # Each taken measurement has a similar simulated measurement pair given translated parameters and simulated parameters
        if source_type == 'csv':
            df = pd.read_csv(source)
            # assuming columns are named in this manner
            v = np.array(df['volt'].tolist())
            i = np.array(df['current'].tolist())

            if type == 'STC':
                a = iv_measurement(voltage=v, current=i, temprature=self.stc_temp,
                                                        irradiance=self.stc_irrad)
                self.ideal_measurement = a
            elif type == 'taken':
                self.measurements.append(iv_measurement(voltage=v, current=i))
        elif source_type == 'measurement':
            if type == 'STC':
                self.ideal_measurement = source
                self.ideal_measurement.temprature = self.stc_temp
                self.ideal_measurement.irradiance = self.stc_irrad
            elif type == 'taken':
                self.measurements.append(source)
        if find_equivalent_circuit:
            if type == 'taken':
                self.measurements[-1].equivalent_circuit(N=self.rows * self.columns)
            elif type == 'STC':
                self.ideal_measurement.equivalent_circuit(N=self.rows * self.columns)
    def characterize(self, index=0, start_guess=2, parameters_are_panel_level=True):
        def reverse(a):
            indices = np.arange(-1, -len(a) - 1, -1)
            b = a[indices]
            return b
        def zigzag_read(mat, no_of_strings):
            colmat = mat.transpose().reshape(1, mat.size)[0]
            col_length = mat.shape[0]
            col_count = mat.shape[1]
            for col in range(1, col_count, 2):
                colmat[col * col_length: (col + 1) * col_length] = reverse(
                    colmat[col * col_length: (col + 1) * col_length])
            colmat = colmat.reshape((no_of_strings, colmat.size // no_of_strings)).transpose()
            return colmat
        # index is the number of measurement within the simulated_measurement property
        current = self.measurements[index].current

        if (self.measurements[index].state_matrix[0].size) > 1:
            # state matrix includes circuit parameters for state_matrix = [rs, rp, iph, idark, ideality, temp]
            rs = np.array(zigzag_read(self.measurements[index].state_matrix[0], self.cell_string_count).transpose())
            rp = np.array(zigzag_read(self.measurements[index].state_matrix[1], self.cell_string_count).transpose())
            iph = np.array(zigzag_read(self.measurements[index].state_matrix[2], self.cell_string_count).transpose())
            idark = np.array(zigzag_read(self.measurements[index].state_matrix[3], self.cell_string_count).transpose())
            ideality = np.array(
                zigzag_read(self.measurements[index].state_matrix[4], self.cell_string_count).transpose())
            temp = np.array(zigzag_read(self.measurements[index].state_matrix[5], self.cell_string_count).transpose())
            isc = np.zeros((self.cell_string_count,))
            for i in range(self.cell_string_count):
                isc[i] = float(iph[i].min())
            order = np.flip(isc.argsort())
            ## Making Similar Isc to the origina
            err = 0.1e-2 # acceptable error
            for i in range(1, len(order)):
                diff = np.abs(isc[order[i]] - isc[order[i - 1]]) / isc[order[i - 1]]
                if diff <= err * 1.2:
                    isc[order[i]] = isc[order[i - 1]] * (1 - err)
                    iph[order[i]] *= (1 - err) ** i
            ##
            V = np.zeros(current.shape)
            I = current
            for z in range(len(isc)):
                m = order[z]
                if z != len(isc) - 1:
                    mask = np.all([(I <= isc[order[z]]), (I >= isc[order[z + 1]])], axis=0)
                else:
                    mask = np.all([(I <= isc[order[z]]), (I >= 0)], axis=0)
                i_zone = I[mask]
                v_zone = np.zeros(i_zone.shape)
                for i in range(z + 1):
                    subpanel = cell_string(
                        rs=rs[m],
                        rp=rp[m],
                        iph=iph[m],
                        idark=idark[m],
                        ideality=ideality[m],
                        temp=temp[m]
                    )
                    subpanel.characterize(current=i_zone, start_guess=start_guess)
                    v_zone += subpanel.voltage
                if (z + 1) <= len(isc) - 1:
                    for i in range(z + 1, len(isc)):
                        self.diode.find_voltage(current=i_zone)
                        v_zone += self.diode.voltage
                V[mask] = v_zone
            V = V[I <= isc[order[0]]]
            I = I[I <= isc[order[0]]]
            V = np.concatenate((V, np.array([0])), axis=0)
            I = np.concatenate((I, np.array([I.max()])), axis=0)
            order = V.argsort()
            V = V[order]
            I = I[order]
            v_new = np.linspace(0, V.max(), 1000)
            i_new = np.interp(v_new, V, I)
            # as the panel is the input measurement has only current values and the state matrix
            self.measurements[index].current = i_new
            self.measurements[index].voltage = v_new
            self.measurements[index].extract_features()
        else:
            if not (parameters_are_panel_level):
                core = np.zeros((self.rows * self.columns, ))
                model_cell = cell_string(
                     rp=self.measurements[index].state_matrix[1] + core,
                     rs=self.measurements[index].state_matrix[0] + core,
                     iph=self.measurements[index].state_matrix[2] + core,
                     temp=self.measurements[index].state_matrix[5] + core,
                     idark=self.measurements[index].state_matrix[3] + core,
                     ideality=self.measurements[index].state_matrix[4] + core,
                     )
                model_cell.characterize(current=current)
                self.measurements[index].current = model_cell.measurement.current
                self.measurements[index].voltage = model_cell.measurement.voltage
            else:
                N = self.rows * self.columns
                rp = self.measurements[index].state_matrix[1]
                rs = self.measurements[index].state_matrix[0]
                Iph = self.measurements[index].state_matrix[2]
                T = self.measurements[index].state_matrix[5]
                Idark = self.measurements[index].state_matrix[3]
                ideality = self.measurements[index].state_matrix[4]
                l = ideality * self.rows * self.columns * k * (T + 273.15) / q
                #---------------- Solving for Voc --------------------
                def func(voc):
                    return Iph - Idark * (np.exp(voc / l) - 1) - voc / rp
                voc = run_optimizer(func, x0=[40], bounds=[(0, 100)], method='TNC')
                voc = voc[0]
                v = np.linspace(0, voc * 1.01, 1000)
                i_produced = np.zeros(v.shape)
                # ------- Do simulation -----------------
                for j in range(len(v)):
                    def sdm(c):
                        # c stands for current
                        return Iph - Idark * (np.exp((v[j] + c * rs) / (l)) - 1) - (
                                v[j] + c * rs) / rp - c

                    c = scipy.optimize.fsolve(sdm, (2))
                    i_produced[j] = c

                v = v[i_produced > 0]
                i_produced = i_produced[i_produced>0]
                self.measurements[index].current = i_produced
                self.measurements[index].voltage = v
            self.measurements[index].rs = rs
            self.measurements[index].rp = rp
            self.measurements[index].iph = Iph
            self.measurements[index].idark = Idark
            self.measurements[index].ideality = ideality
            self.measurements[index].extract_features()
    def translate_parameters(self, G, T):
        # G is irradiance in W/m^2 and T is temeperature in celesius
        if self.ideal_measurement.rs == []:
            self.ideal_measurement.equivalent_circuit(N=self.rows * self.columns)
        if self.ideal_measurement.voc == []:
            self.ideal_measurement.extract_features()

        voc_stc = self.ideal_measurement.voc
        isc_stc = self.ideal_measurement.isc
        rs_stc = self.ideal_measurement.rs
        rp_stc = self.ideal_measurement.rp
        ideality_stc = self.ideal_measurement.ideality
        mu_isc = self.isc_thermal_coeffcient
        mu_voc = self.voc_thermal_coeffcient
        g_stc = self.stc_irrad
        t_stc = self.stc_temp + 273.15
        #---------------- Starting parameter translation
        # G is target irradiance and T is temperature
        T = T + 273.15
        isc = (G / g_stc) ** self.isc_nonlinearity_factor * (isc_stc + mu_isc * (T - t_stc))
        # ----------- Calculating Voc ---------------------------------------------------
        # c1 = 5.468511e-2
        # c2 = 5.973869e-3
        # c3 = 7.616178e-4
        # voc = voc_stc + c1 * np.log(G / g_stc) + c2 * np.log(G / g_stc) ** 2 + c3 * np.log(G / g_stc) ** 3 + mu_voc * (
        #             T - t_stc)
        # using the panel specific correction factors
        c = self.voc_irradiance_correction_factors
        voc = voc_stc + mu_voc * (T - t_stc)
        h = np.log(G/g_stc)
        for i in range(len(c)):
            voc += c[i] * h ** i
        # -------------- Calculating Rs and Rp ---------------------------------
        # 0.217
        rs = T / t_stc * (1 - 0.217 * np.log(G / g_stc)) * rs_stc
        # rs = g_stc / G * rs_stc
        rp = g_stc / G * rp_stc
        # changing the ideality factor---------------------------------------
        n = ideality_stc * (T / t_stc)
        '''source for change of ideality
        https://www.sciencedirect.com/science/article/abs/pii/S1364032118300984?casa_token=h29icoynxasAAAAA:BzsRQ-caCSVfpyw4OFYyqDBoyKTQjr1_qFnv2u5V6_xg3x_gJhL8kZlA1dhUIk-_EdelatxzDA
        '''
        # --------------- Calculating the dark current ------------------------
        # l = n * self.rows * self.columns * k * T / q
        # A = np.exp((voc_stc + mu_voc * (T - t_stc) + l * np.log(G)) / l)
        # B = np.exp((isc_stc + mu_isc * (T - t_stc)) * rs / l)
        # numerator = (1 + rs / rp) * (isc_stc + mu_isc * (T - t_stc)) -(voc_stc + mu_voc * (T - t_stc) + l * np.log(G))/rp
        # idark = numerator / (A - B)
        # # ---------------- Calculating the photocurrent
        # iph = idark * (np.exp(voc / l) - 1) + voc / rp

        #--------------- We use another analytical way of estimating the photocurrent and the saturation current
        l = n * self.rows * self.columns * k * T / q
        A = np.exp(voc / l) - 1
        B = np.exp(isc * rs / l) - 1
        C = voc / rp
        D = isc * rs / rp
        idark = (isc - (C - D)) / (A - B)
        iph = idark * A + C

        v = np.linspace(0, voc * 1.01, 1000)
        i_produced = np.zeros(v.shape)
        # ------- Do simulation -----------------
        for j in range(len(v)):
            def sdm(c):
                # c stands for current
                return iph - idark * (np.exp((v[j] + c * rs) / (l)) - 1) - (
                        v[j] + c * rs) / rp - c

            c = scipy.optimize.fsolve(sdm, (2))
            i_produced[j] = c

        v = v[i_produced > 0]
        i_produced = i_produced[i_produced > 0]
        self.translated_measurement = iv_measurement(voltage=v, current=i_produced)
        self.translated_measurement.extract_features()
        self.translated_measurement.rs = rs
        self.translated_measurement.rp = rp
        self.translated_measurement.idark = idark
        self.translated_measurement.iph = iph
        self.translated_measurement.ideality = n
