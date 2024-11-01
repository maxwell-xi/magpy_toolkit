import numpy as np
import pandas as pd
from data_visualization_and_processing import helper
import glob
import os
import zipfile
from scipy import signal
import scipy.constants as sc

def search_spectrum_peaks(spectra_df, quantity='h_field', max_value_used=True, low_freq_truncated=True, high_freq_truncated=True, output_rounded=True):
    # remove artificial high fields at the low and high frequency ends
    if low_freq_truncated == True:
        spectra_df = spectra_df[spectra_df['frequency']>=1e4] # 8e4 may be needed for wireless chargers 
        
    if high_freq_truncated == True:
        spectra_df = spectra_df[spectra_df['frequency']<=4e6]
    
    # add normalized values (in dB) of the fields
    spectra_df['h_field_rms_total_n_db'] = 20*np.log10(spectra_df['h_field_rms_total']/np.max(spectra_df['h_field_rms_total']))
    spectra_df['e_field_rms_total_n_db'] = 20*np.log10(spectra_df['e_field_rms_total']/np.max(spectra_df['e_field_rms_total']))
    spectra_df['h_field_rms_max_n_db'] = 20*np.log10(spectra_df['h_field_rms_max']/np.max(spectra_df['h_field_rms_max']))
    spectra_df['e_field_rms_max_n_db'] = 20*np.log10(spectra_df['e_field_rms_max']/np.max(spectra_df['e_field_rms_max']))
    
    # extract the fundamental frequency based on H-field, considering we are evaluating H-field sources   
    if max_value_used == True:
        f0 = spectra_df['frequency'].loc[np.argmax(spectra_df['h_field_rms_max_n_db'])] 
    else:
        f0 = spectra_df['frequency'].loc[np.argmax(spectra_df['h_field_rms_total_n_db'])] 
    
    # ignore peaks below the fundamental frequency
    spectra_df = spectra_df[spectra_df['frequency']>=f0]
    
    # exclude low fields to facilitate peak search
    if quantity == 'h_field':
        if max_value_used == True:
            harmonics_df = spectra_df[spectra_df['h_field_rms_max_n_db']>-30]
        else:
            harmonics_df = spectra_df[spectra_df['h_field_rms_total_n_db']>-30]
    elif quantity == 'e_field':
        if max_value_used == True:
            harmonics_df = spectra_df[spectra_df['e_field_rms_max_n_db']>-30] # a high threshold used to filter ripples
        else:
            harmonics_df = spectra_df[spectra_df['e_field_rms_total_n_db']>-30] # a high threshold used to filter ripples
    else:
        print('Undefined quantity!')    
    
    # create data holders
    f = np.array(harmonics_df['frequency']) # convert pandas series to numpy array
    h_tot_db = np.array(harmonics_df['h_field_rms_total_n_db'])
    h_tot = np.array(harmonics_df['h_field_rms_total'])    
    h_max_db = np.array(harmonics_df['h_field_rms_max_n_db'])
    h_max = np.array(harmonics_df['h_field_rms_max'])
    e_tot_db = np.array(harmonics_df['e_field_rms_total_n_db'])
    e_tot = np.array(harmonics_df['e_field_rms_total'])
    e_max_db = np.array(harmonics_df['e_field_rms_max_n_db'])
    e_max = np.array(harmonics_df['e_field_rms_max'])
    
    f_step = np.max(np.diff(spectra_df['frequency']))
    f_diff = np.diff(f)
    segment_num = np.size(f_diff[f_diff>f_step]) + 1
        
    f_segments = [ [] for _ in range(segment_num) ]
    h_tot_db_segments = [ [] for _ in range(segment_num) ]
    h_tot_segments = [ [] for _ in range(segment_num) ]
    h_max_db_segments = [ [] for _ in range(segment_num) ]
    h_max_segments = [ [] for _ in range(segment_num) ]
    e_tot_db_segments = [ [] for _ in range(segment_num) ]
    e_tot_segments = [ [] for _ in range(segment_num) ]
    e_max_db_segments = [ [] for _ in range(segment_num) ]
    e_max_segments = [ [] for _ in range(segment_num) ]
    
    f_segments[0].append(f[0])
    h_tot_db_segments[0].append(h_tot_db[0])
    h_tot_segments[0].append(h_tot[0])
    h_max_db_segments[0].append(h_max_db[0])
    h_max_segments[0].append(h_max[0])
    e_tot_db_segments[0].append(e_tot_db[0])
    e_tot_segments[0].append(e_tot[0])
    e_max_db_segments[0].append(e_max_db[0])
    e_max_segments[0].append(e_max[0])
    
    # split the data into multiple non-continuous segments across the frequency axis
    j = 0
    for i in range(len(f_diff)):
        if f_diff[i] > f_step:
            j = j + 1
        f_segments[j].append(f[i+1])
        h_tot_db_segments[j].append(h_tot_db[i+1])
        h_tot_segments[j].append(h_tot[i+1])
        h_max_db_segments[j].append(h_max_db[i+1])
        h_max_segments[j].append(h_max[i+1])
        e_tot_db_segments[j].append(e_tot_db[i+1])
        e_tot_segments[j].append(e_tot[i+1]) 
        e_max_db_segments[j].append(e_max_db[i+1])
        e_max_segments[j].append(e_max[i+1])    
    
    # create data holders
    f_peaks = []
    h_tot_db_peaks = []
    h_tot_peaks = []
    h_max_db_peaks = []
    h_max_peaks = []
    e_tot_db_peaks = []
    e_tot_peaks = []
    e_max_db_peaks = []
    e_max_peaks = []

    # extract the peak fields of all segments and the corresponding frequencies    
    if quantity == 'h_field':
        if max_value_used == True:
            h_max_db_peaks = [np.max(x) for x in h_max_db_segments]
            target_segments = h_max_db_segments
            for i in range(len(f_segments)):
                f_peaks.append(f_segments[i][np.argmax(target_segments[i])])
                h_tot_db_peaks.append(h_tot_db_segments[i][np.argmax(target_segments[i])])
                h_tot_peaks.append(h_tot_segments[i][np.argmax(target_segments[i])])
                h_max_peaks.append(h_max_segments[i][np.argmax(target_segments[i])])
                e_tot_db_peaks.append(e_tot_db_segments[i][np.argmax(target_segments[i])])
                e_tot_peaks.append(e_tot_segments[i][np.argmax(target_segments[i])]) 
                e_max_db_peaks.append(e_max_db_segments[i][np.argmax(target_segments[i])])
                e_max_peaks.append(e_max_segments[i][np.argmax(target_segments[i])])
        else:    
            h_tot_db_peaks = [np.max(x) for x in h_tot_db_segments]
            target_segments = h_tot_db_segments
            for i in range(len(f_segments)):
                f_peaks.append(f_segments[i][np.argmax(target_segments[i])])
                h_tot_peaks.append(h_tot_segments[i][np.argmax(target_segments[i])])
                h_max_db_peaks.append(h_max_db_segments[i][np.argmax(target_segments[i])])
                h_max_peaks.append(h_max_segments[i][np.argmax(target_segments[i])])
                e_tot_db_peaks.append(e_tot_db_segments[i][np.argmax(target_segments[i])])
                e_tot_peaks.append(e_tot_segments[i][np.argmax(target_segments[i])])
                e_max_db_peaks.append(e_max_db_segments[i][np.argmax(target_segments[i])])
                e_max_peaks.append(e_max_segments[i][np.argmax(target_segments[i])])
    elif quantity == 'e_field':
        if max_value_used == True:
            e_max_db_peaks = [np.max(x) for x in e_max_db_segments]
            target_segments = e_max_db_segments
            for i in range(len(f_segments)):
                f_peaks.append(f_segments[i][np.argmax(target_segments[i])])
                h_tot_db_peaks.append(h_tot_db_segments[i][np.argmax(target_segments[i])])
                h_tot_peaks.append(h_tot_segments[i][np.argmax(target_segments[i])])
                h_max_db_peaks.append(h_max_db_segments[i][np.argmax(target_segments[i])])
                h_max_peaks.append(h_max_segments[i][np.argmax(target_segments[i])])
                e_tot_db_peaks.append(e_tot_db_segments[i][np.argmax(target_segments[i])])
                e_tot_peaks.append(e_tot_segments[i][np.argmax(target_segments[i])]) 
                e_max_peaks.append(e_max_segments[i][np.argmax(target_segments[i])])       
        else:
            e_tot_db_peaks = [np.max(x) for x in e_tot_db_segments]
            target_segments = e_tot_db_segments
            for i in range(len(f_segments)):
                f_peaks.append(f_segments[i][np.argmax(target_segments[i])])
                h_tot_db_peaks.append(h_tot_db_segments[i][np.argmax(target_segments[i])])
                h_tot_peaks.append(h_tot_segments[i][np.argmax(target_segments[i])])
                h_max_db_peaks.append(h_max_db_segments[i][np.argmax(target_segments[i])])
                h_max_peaks.append(h_max_segments[i][np.argmax(target_segments[i])])                
                e_tot_peaks.append(e_tot_segments[i][np.argmax(target_segments[i])])
                e_max_db_peaks.append(e_max_db_segments[i][np.argmax(target_segments[i])])
                e_max_peaks.append(e_max_segments[i][np.argmax(target_segments[i])])
    else:
        print('Undefined quantity!') 
        
    peaks_df = pd.DataFrame([])
    peaks_df['Freq [kHz]'] = np.array(f_peaks)/1e3
    peaks_df['Freq [f0]'] = np.array(f_peaks)/f0    
    peaks_df['Htot_normalized [dB]'] = h_tot_db_peaks
    peaks_df['Htot [A/m]'] = h_tot_peaks
    peaks_df['Hmax_normalized [dB]'] = h_max_db_peaks
    peaks_df['Hmax [A/m]'] = h_max_peaks
    peaks_df['Etot_normalized [dB]'] = e_tot_db_peaks
    peaks_df['Etot [V/m]'] = e_tot_peaks
    peaks_df['Emax_normalized [dB]'] = e_max_db_peaks
    peaks_df['Emax [V/m]'] = e_max_peaks
    
    if output_rounded == True: # rounded for a clean display
        peaks_df['Freq [kHz]'] = peaks_df['Freq [kHz]'].round(decimals=1)
        peaks_df['Freq [f0]'] = peaks_df['Freq [f0]'].round(decimals=1)    
        peaks_df['Htot_normalized [dB]'] = peaks_df['Htot_normalized [dB]'].round(decimals=1)
        peaks_df['Htot [A/m]'] = peaks_df['Htot [A/m]'].round(decimals=3) 
        peaks_df['Hmax_normalized [dB]'] = peaks_df['Hmax_normalized [dB]'].round(decimals=1)
        peaks_df['Hmax [A/m]'] = peaks_df['Hmax [A/m]'].round(decimals=3)
        peaks_df['Etot_normalized [dB]'] = peaks_df['Etot_normalized [dB]'].round(decimals=1)
        peaks_df['Etot [V/m]'] = peaks_df['Etot [V/m]'].round(decimals=3)
        peaks_df['Emax_normalized [dB]'] = peaks_df['Emax_normalized [dB]'].round(decimals=1)
        peaks_df['Emax [V/m]'] = peaks_df['Emax [V/m]'].round(decimals=3)
          
    return peaks_df, f0

def multi_frequency_compliance_evaluation(input_df, quantity='h_field', max_value_used=True):
    if quantity == 'h_field':
        limit = 21 # ICNIRP 2010, electrical stimulation limit    
            
        if max_value_used == True:
            ratios = np.array(input_df['Hmax [A/m]'])/limit
        else:
            ratios = np.array(input_df['Htot [A/m]'])/limit
    elif quantity == 'e_field':
        limit = 83 # ICNIRP 2010, electrical stimulation limit    
            
        if max_value_used == True:
            ratios = np.array(input_df['Emax [V/m]'])/limit
        else:
            ratios = np.array(input_df['Etot [V/m]'])/limit
    else:
        print('Undefined quantity!') 
        
    exposure_ratio = np.sum(ratios)
    
    return exposure_ratio
    
def extrapolation_factor_fitted(g_n, local_field_at_probe_tip=True):
    #coeff = np.array([1, 1, -1.04, 11.0, -31.7, 45.9, -32.2, 8.95]) # derived based on 1 cm2 avg. total field at probe tip with least square method
    coeff_avg = np.array([1, 1, -1.01, 15.9, -50.8, 74.7, -51.4, 13.7]) # derived based on 1 cm2 avg. total field at probe tip with 97.5th percentile quantile regression 
    coeff_local = np.array([1, 1, -0.764, 14.5, -47.7, 70.7, -48.7, 13.1]) # derived based on local total field at probe tip with 97.5th percentile quantile regression
    g_d_product = g_n * 18.5e-3
    item = np.array([1, g_d_product, g_d_product**2, g_d_product**3, g_d_product**4, g_d_product**5, g_d_product**6, g_d_product**7])

    if local_field_at_probe_tip == True:
        extrap_factor = np.sum(coeff_local * item)
    else:
        extrap_factor = np.sum(coeff_avg * item)
    
    return extrap_factor

def mimic_magpy_probe(field, grid_mm, probe_center_loc_mm = [0, 0, 18.5], sensor_avg_considered = False): 
    '''
    INPUT: H-field data (incl. x-, y-, z-components and the total field), the corresponding grid (0.5 mm step), and the coordinates of the probe center
    OUTPUT: H-field results (measured value and true value) at the probe center and the probe tip, following the implementation of MAGPy V2.x 
    ''' 

    # extract H-field at probe center
    i_center = np.argwhere(np.isclose(grid_mm[0], probe_center_loc_mm[0]))[0,0]
    j_center = np.argwhere(np.isclose(grid_mm[1], probe_center_loc_mm[1]))[0,0]
    k_center = np.argwhere(np.isclose(grid_mm[2], probe_center_loc_mm[2]))[0,0]
    ht_center_true = field[3][i_center, j_center, k_center] # total local H-field at the probe center with the indices (i_center, j_center, k_center)
    
    # derive measured H-fields at probe center by averaging over 8 sensors 
    i_plus = i_center + 22; i_minus = i_center - 22 # 22 grid lines corresponds to 11 mm
    j_plus = j_center + 22; j_minus = j_center - 22
    k_top = k_center + 22; k_bottom = k_center - 22  
    
    i_sensor = [i_plus, i_minus, i_minus, i_plus, i_plus, i_minus, i_minus, i_plus]
    j_sensor = [j_plus, j_plus, j_minus, j_minus, j_plus, j_plus, j_minus, j_minus]
    k_sensor = [k_bottom, k_bottom, k_bottom, k_bottom, k_top, k_top, k_top, k_top]
    
    hx_sensor = []; hy_sensor = []; hz_sensor = []; ht_sensor = []
    if sensor_avg_considered == True:
        for n in np.arange(8):
            hx_sensor_temp = helper.x_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
            hy_sensor_temp = helper.y_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
            hz_sensor_temp = helper.z_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
            ht_sensor_temp = np.sqrt(hx_sensor_temp**2 + hy_sensor_temp**2 + hz_sensor_temp**2)
            hx_sensor.append(hx_sensor_temp)
            hy_sensor.append(hy_sensor_temp)
            hz_sensor.append(hz_sensor_temp)
            ht_sensor.append(ht_sensor_temp)
    else:
        for n in np.arange(8):
            hx_sensor.append(field[0][i_sensor[n], j_sensor[n], k_sensor[n]])
            hy_sensor.append(field[1][i_sensor[n], j_sensor[n], k_sensor[n]])
            hz_sensor.append(field[2][i_sensor[n], j_sensor[n], k_sensor[n]])
            ht_sensor.append(field[3][i_sensor[n], j_sensor[n], k_sensor[n]])

    hx_center = np.mean(hx_sensor); hy_center = np.mean(hy_sensor); hz_center = np.mean(hz_sensor); ht_center = np.mean(ht_sensor)
    ht_center_combined = np.sqrt(hx_center**2 + hy_center**2 + hz_center**2)
    
    ht_center_result = [ht_center, ht_center_combined, ht_center_true] # confirmed by Dmytro on 2024-3-11 that ht_center was used to derive the normalized gradient at probe center

    # extract gradient at probe center
    gz_center_true = (field[3][i_center, j_center, k_center-1] - field[3][i_center, j_center, k_center+1]) / 1e-3
    
    # derive measured gradients at probe center with a simplifed approach (instead of the optimization approach used in MAGPy)
    gz_1 = (ht_sensor[0] - ht_sensor[4]) / 22e-3 
    gz_2 = (ht_sensor[1] - ht_sensor[5]) / 22e-3
    gz_3 = (ht_sensor[2] - ht_sensor[6]) / 22e-3
    gz_4 = (ht_sensor[3] - ht_sensor[7]) / 22e-3    
    gz_center = np.mean([gz_1, gz_2, gz_3, gz_4]) 
    
    gz_center_result = [gz_center, gz_center_true]
    
    # gradients along x and y not exposed
    #gx_1 = (ht_sensor[1] - ht_sensor[0]) / 22e-3 
    #gx_2 = (ht_sensor[5] - ht_sensor[4]) / 22e-3
    #gx_3 = (ht_sensor[6] - ht_sensor[7]) / 22e-3
    #gx_4 = (ht_sensor[2] - ht_sensor[3]) / 22e-3    
    #gx_center = np.mean([gx_1, gx_2, gx_3, gx_4]) 
    
    #gy_1 = (ht_sensor[3] - ht_sensor[0]) / 22e-3 
    #gy_2 = (ht_sensor[7] - ht_sensor[4]) / 22e-3
    #gy_3 = (ht_sensor[6] - ht_sensor[5]) / 22e-3
    #gy_4 = (ht_sensor[2] - ht_sensor[1]) / 22e-3    
    #gy_center = np.mean([gy_1, gy_2, gy_3, gy_4])
    
    #gt_center = np.sqrt(gx_center**2 + gy_center**2 + gz_center**2)
    #g_center = [gx_center, gy_center, gz_center, gt_center]
        
    # extract H-field at probe tip
    k_tip = k_center - 37  # 37 grid lines corresponds to 18.5 mm
    ht_tip_true = field[3][i_center, j_center, k_tip] # total local H-field at the probe tip which is 18.5 mm below the probe center  
  
    # extract H-fields at the four projection points on the probe surface corresponding to the four vertical sensor pairs
    ht_tip_true_1 = field[3][i_plus, j_plus, k_tip]
    ht_tip_true_2 = field[3][i_minus, j_plus, k_tip]
    ht_tip_true_3 = field[3][i_minus, j_minus, k_tip]
    ht_tip_true_4 = field[3][i_plus, j_minus, k_tip]
    
    # derive measured H-fields at the four projection points by field extrapolation
    ht_mid_1 = (ht_sensor[0] + ht_sensor[4]) / 2 
    ht_mid_2 = (ht_sensor[1] + ht_sensor[5]) / 2
    ht_mid_3 = (ht_sensor[2] + ht_sensor[6]) / 2
    ht_mid_4 = (ht_sensor[3] + ht_sensor[7]) / 2    
    gz_n_1 = gz_1 / ht_mid_1; gz_n_2 = gz_2 / ht_mid_2; gz_n_3 = gz_3 / ht_mid_3; gz_n_4 = gz_4 / ht_mid_4    
    ht_tip_1 = ht_mid_1 * extrapolation_factor_fitted(gz_n_1)
    ht_tip_2 = ht_mid_2 * extrapolation_factor_fitted(gz_n_2)
    ht_tip_3 = ht_mid_3 * extrapolation_factor_fitted(gz_n_3)
    ht_tip_4 = ht_mid_4 * extrapolation_factor_fitted(gz_n_4)
    
    ht_tip_result = [ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4, ht_tip_true_1, ht_tip_true_2, ht_tip_true_3, ht_tip_true_4, ht_tip_true]
   
    return ht_center_result, gz_center_result, ht_tip_result
    
def mimic_old_magpy_probe(field, grid_mm, probe_center_loc_mm = [0, 0, 29.5]):    
    i_center = np.argwhere(grid_mm[0] == probe_center_loc_mm[0])[0,0]
    j_center = np.argwhere(grid_mm[1] == probe_center_loc_mm[1])[0,0]
    k_center = np.argwhere(grid_mm[2] == probe_center_loc_mm[2])[0,0]
    ht_center_true = field[3][i_center, j_center, k_center]
    
    k_tip = k_center - 59  # 59 grid lines corresponds to 29.5 mm
    ht_tip_true = field[3][i_center, j_center, k_tip]
    
    i_plus = i_center + 20; i_minus = i_center - 20
    j_plus = j_center + 20; j_minus = j_center - 20
    k_top = k_center + 20; k_bottom = k_center - 20    
  
    ht_tip_true = field[3][i_center, j_center, k_tip]
        
    i_sensor = [i_plus, i_minus, i_minus, i_plus, i_plus, i_minus, i_minus, i_plus]
    j_sensor = [j_plus, j_plus, j_minus, j_minus, j_plus, j_plus, j_minus, j_minus]
    k_sensor = [k_bottom, k_bottom, k_bottom, k_bottom, k_top, k_top, k_top, k_top]
    
    hx_sensor = []; hy_sensor = []; hz_sensor = []; ht_sensor = []
    for n in np.arange(8):
        hx_sensor.append(field[0][i_sensor[n], j_sensor[n], k_sensor[n]])
        hy_sensor.append(field[1][i_sensor[n], j_sensor[n], k_sensor[n]])
        hz_sensor.append(field[2][i_sensor[n], j_sensor[n], k_sensor[n]])
        ht_sensor.append(field[3][i_sensor[n], j_sensor[n], k_sensor[n]])
        
    hx_center = np.mean(hx_sensor); hy_center = np.mean(hy_sensor); hz_center = np.mean(hz_sensor); ht_center = np.mean(ht_sensor)
    ht_center_combined = np.sqrt(hx_center**2 + hy_center**2 + hz_center**2)
    h_center = [hx_center, hy_center, hz_center, ht_center_combined]
    h_center_rms = [x/np.sqrt(2) for x in h_center]
    ht_center_error = 20*np.log10(ht_center_combined / ht_center_true)
    
    gz_1 = (ht_sensor[0] - ht_sensor[4]) / 20e-3 
    gz_2 = (ht_sensor[1] - ht_sensor[5]) / 20e-3
    gz_3 = (ht_sensor[2] - ht_sensor[6]) / 20e-3
    gz_4 = (ht_sensor[3] - ht_sensor[7]) / 20e-3
    
    gz_n_center = np.mean([gz_1, gz_2, gz_3, gz_4]) / ht_center
    
    gx_1 = (ht_sensor[1] - ht_sensor[0]) / 20e-3 
    gx_2 = (ht_sensor[5] - ht_sensor[4]) / 20e-3
    gx_3 = (ht_sensor[6] - ht_sensor[7]) / 20e-3
    gx_4 = (ht_sensor[2] - ht_sensor[3]) / 20e-3
    
    gx_n_center = np.mean([gx_1, gx_2, gx_3, gx_4]) / ht_center
    
    gy_1 = (ht_sensor[3] - ht_sensor[0]) / 20e-3 
    gy_2 = (ht_sensor[7] - ht_sensor[4]) / 20e-3
    gy_3 = (ht_sensor[6] - ht_sensor[5]) / 20e-3
    gy_4 = (ht_sensor[2] - ht_sensor[1]) / 20e-3
    
    gy_n_center = np.mean([gy_1, gy_2, gy_3, gy_4]) / ht_center
    
    gt_n_center = np.sqrt(gx_n_center**2 + gy_n_center**2 + gz_n_center**2)
    g_n_center = [gx_n_center, gy_n_center, gz_n_center, gt_n_center]        
    
    ht_tip = ht_center_combined * np.exp(2*gz_n_center*29.5e-3)
    ht_tip_rms = ht_tip/np.sqrt(2)
    
    ht_tip_error = 20*np.log10(ht_tip / ht_tip_true) 
    
    return h_center_rms, ht_center_error, g_n_center, ht_tip_rms, ht_tip_error
    
def mimic_magpy_probe_with_sensor_avg(field, grid_mm, probe_center_loc_mm = [0, 0, 18.5]):    
    i_center = np.argwhere(np.isclose(grid_mm[0], probe_center_loc_mm[0]))[0,0]
    j_center = np.argwhere(np.isclose(grid_mm[1], probe_center_loc_mm[1]))[0,0]
    k_center = np.argwhere(np.isclose(grid_mm[2], probe_center_loc_mm[2]))[0,0]
    hx_center_true = helper.x_component_avg(field, 20, i_center, j_center, k_center) # 10 mm sensor sidelength corresponds to 20 grid lines
    hy_center_true = helper.y_component_avg(field, 20, i_center, j_center, k_center)
    hz_center_true = helper.z_component_avg(field, 20, i_center, j_center, k_center)
    ht_center_true = np.sqrt(hx_center_true**2 + hy_center_true**2 + hz_center_true**2)
    
    k_tip = k_center - 37  # 37 grid lines corresponds to 18.5 mm
    hx_tip_true = helper.x_component_avg(field, 20, i_center, j_center, k_tip) 
    hy_tip_true = helper.y_component_avg(field, 20, i_center, j_center, k_tip)
    hz_tip_true = helper.z_component_avg(field, 20, i_center, j_center, k_tip)    
    ht_tip_true = np.sqrt(hx_tip_true**2 + hy_tip_true**2 + hz_tip_true**2)
    
    i_plus = i_center + 22; i_minus = i_center - 22
    j_plus = j_center + 22; j_minus = j_center - 22
    k_top = k_center + 22; k_bottom = k_center - 22 
    
    hx_tip_true_1 = helper.x_component_avg(field, 20, i_plus, j_plus, k_tip) 
    hy_tip_true_1 = helper.y_component_avg(field, 20, i_plus, j_plus, k_tip)
    hz_tip_true_1 = helper.z_component_avg(field, 20, i_plus, j_plus, k_tip)    
    ht_tip_true_1 = np.sqrt(hx_tip_true_1**2 + hy_tip_true_1**2 + hz_tip_true_1**2)
    
    hx_tip_true_2 = helper.x_component_avg(field, 20, i_minus, j_plus, k_tip) 
    hy_tip_true_2 = helper.y_component_avg(field, 20, i_minus, j_plus, k_tip)
    hz_tip_true_2 = helper.z_component_avg(field, 20, i_minus, j_plus, k_tip)    
    ht_tip_true_2 = np.sqrt(hx_tip_true_2**2 + hy_tip_true_2**2 + hz_tip_true_2**2)
    
    hx_tip_true_3 = helper.x_component_avg(field, 20, i_minus, j_minus, k_tip) 
    hy_tip_true_3 = helper.y_component_avg(field, 20, i_minus, j_minus, k_tip)
    hz_tip_true_3 = helper.z_component_avg(field, 20, i_minus, j_minus, k_tip)    
    ht_tip_true_3 = np.sqrt(hx_tip_true_3**2 + hy_tip_true_3**2 + hz_tip_true_3**2)
    
    hx_tip_true_4 = helper.x_component_avg(field, 20, i_plus, j_minus, k_tip) 
    hy_tip_true_4 = helper.y_component_avg(field, 20, i_plus, j_minus, k_tip)
    hz_tip_true_4 = helper.z_component_avg(field, 20, i_plus, j_minus, k_tip)    
    ht_tip_true_4 = np.sqrt(hx_tip_true_4**2 + hy_tip_true_4**2 + hz_tip_true_4**2)
    
    i_sensor = [i_plus, i_minus, i_minus, i_plus, i_plus, i_minus, i_minus, i_plus]
    j_sensor = [j_plus, j_plus, j_minus, j_minus, j_plus, j_plus, j_minus, j_minus]
    k_sensor = [k_bottom, k_bottom, k_bottom, k_bottom, k_top, k_top, k_top, k_top]
    
    hx_sensor = []; hy_sensor = []; hz_sensor = []; ht_sensor = []
    for n in np.arange(8):
        hx_sensor_temp = helper.x_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
        hy_sensor_temp = helper.y_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
        hz_sensor_temp = helper.z_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
        hx_sensor.append(hx_sensor_temp)
        hy_sensor.append(hy_sensor_temp)
        hz_sensor.append(hz_sensor_temp)
        ht_sensor.append(np.sqrt(hx_sensor_temp**2 + hy_sensor_temp**2 + hz_sensor_temp**2))
        
    hx_center = np.mean(hx_sensor); hy_center = np.mean(hy_sensor); hz_center = np.mean(hz_sensor); ht_center = np.mean(ht_sensor)
    ht_center_combined = np.sqrt(hx_center**2 + hy_center**2 + hz_center**2)
    h_center = [hx_center, hy_center, hz_center, ht_center_combined]
    h_center_rms = [x/np.sqrt(2) for x in h_center]
    ht_center_error = 20*np.log10(ht_center_combined / ht_center_true)
    
    gz_1 = (ht_sensor[0] - ht_sensor[4]) / 22e-3 
    gz_2 = (ht_sensor[1] - ht_sensor[5]) / 22e-3
    gz_3 = (ht_sensor[2] - ht_sensor[6]) / 22e-3
    gz_4 = (ht_sensor[3] - ht_sensor[7]) / 22e-3
    
    gz_n_center = np.mean([gz_1, gz_2, gz_3, gz_4]) / ht_center
    
    gx_1 = (ht_sensor[1] - ht_sensor[0]) / 22e-3 
    gx_2 = (ht_sensor[5] - ht_sensor[4]) / 22e-3
    gx_3 = (ht_sensor[6] - ht_sensor[7]) / 22e-3
    gx_4 = (ht_sensor[2] - ht_sensor[3]) / 22e-3
    
    gx_n_center = np.mean([gx_1, gx_2, gx_3, gx_4]) / ht_center
    
    gy_1 = (ht_sensor[3] - ht_sensor[0]) / 22e-3 
    gy_2 = (ht_sensor[7] - ht_sensor[4]) / 22e-3
    gy_3 = (ht_sensor[6] - ht_sensor[5]) / 22e-3
    gy_4 = (ht_sensor[2] - ht_sensor[1]) / 22e-3
    
    gy_n_center = np.mean([gy_1, gy_2, gy_3, gy_4]) / ht_center
    gt_n_center = np.sqrt(gx_n_center**2 + gy_n_center**2 + gz_n_center**2)
    g_n_center = [gx_n_center, gy_n_center, gz_n_center, gt_n_center]
    
    ht_mid_1 = (ht_sensor[0] + ht_sensor[4]) / 2 
    ht_mid_2 = (ht_sensor[1] + ht_sensor[5]) / 2
    ht_mid_3 = (ht_sensor[2] + ht_sensor[6]) / 2
    ht_mid_4 = (ht_sensor[3] + ht_sensor[7]) / 2
    
    gz_n_1 = gz_1 / ht_mid_1; gz_n_2 = gz_2 / ht_mid_2; gz_n_3 = gz_3 / ht_mid_3; gz_n_4 = gz_4 / ht_mid_4     
    
    ht_tip_1 = ht_mid_1 * extrapolation_factor_fitted(gz_n_1)
    ht_tip_2 = ht_mid_2 * extrapolation_factor_fitted(gz_n_2)
    ht_tip_3 = ht_mid_3 * extrapolation_factor_fitted(gz_n_3)
    ht_tip_4 = ht_mid_4 * extrapolation_factor_fitted(gz_n_4)
    ht_tip_max = np.max([ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4])
    ht_tip_avg = np.mean([ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4])
    ht_tip = [ht_tip_max, ht_tip_avg]
    ht_tip_rms = [x/np.sqrt(2) for x in ht_tip]    
    
    ht_tip_error_max = 20*np.log10(ht_tip_max / np.max([ht_tip_true_1, ht_tip_true_2, ht_tip_true_3, ht_tip_true_4]))
    ht_tip_error_avg = 20*np.log10(ht_tip_avg / np.mean([ht_tip_true_1, ht_tip_true_2, ht_tip_true_3, ht_tip_true_4])) 
    ht_tip_error_true_avg = 20*np.log10(ht_tip_avg / ht_tip_true)
    ht_tip_error = [ht_tip_error_max, ht_tip_error_avg, ht_tip_error_true_avg]    
    
    return h_center_rms, ht_center_error, g_n_center, ht_tip_rms, ht_tip_error  

def mimic_magpy_probe_with_sensor_avg_simplified(field, grid_mm, probe_center_loc_mm = [0, 0, 18.5]):    
    i_center = np.argwhere(np.isclose(grid_mm[0], probe_center_loc_mm[0]))[0,0]
    j_center = np.argwhere(np.isclose(grid_mm[1], probe_center_loc_mm[1]))[0,0]
    k_center = np.argwhere(np.isclose(grid_mm[2], probe_center_loc_mm[2]))[0,0]
       
    k_tip = k_center - 37  # 37 grid lines corresponds to 18.5 mm
    
    i_plus = i_center + 22; i_minus = i_center - 22
    j_plus = j_center + 22; j_minus = j_center - 22
    k_top = k_center + 22; k_bottom = k_center - 22 
   
    i_sensor = [i_plus, i_minus, i_minus, i_plus, i_plus, i_minus, i_minus, i_plus]
    j_sensor = [j_plus, j_plus, j_minus, j_minus, j_plus, j_plus, j_minus, j_minus]
    k_sensor = [k_bottom, k_bottom, k_bottom, k_bottom, k_top, k_top, k_top, k_top]
    
    hx_sensor = []; hy_sensor = []; hz_sensor = []; ht_sensor = []
    for n in np.arange(8):
        hx_sensor_temp = helper.x_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
        hy_sensor_temp = helper.y_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
        hz_sensor_temp = helper.z_component_avg(field, 20, i_sensor[n], j_sensor[n], k_sensor[n])
        hx_sensor.append(hx_sensor_temp)
        hy_sensor.append(hy_sensor_temp)
        hz_sensor.append(hz_sensor_temp)
        ht_sensor.append(np.sqrt(hx_sensor_temp**2 + hy_sensor_temp**2 + hz_sensor_temp**2))
        
    hx_center = np.mean(hx_sensor); hy_center = np.mean(hy_sensor); hz_center = np.mean(hz_sensor); ht_center = np.mean(ht_sensor)
    ht_center_combined = np.sqrt(hx_center**2 + hy_center**2 + hz_center**2)
    h_center = [hx_center, hy_center, hz_center, ht_center_combined]
    h_center_rms = [x/np.sqrt(2) for x in h_center]
    
    gz_1 = (ht_sensor[0] - ht_sensor[4]) / 22e-3 
    gz_2 = (ht_sensor[1] - ht_sensor[5]) / 22e-3
    gz_3 = (ht_sensor[2] - ht_sensor[6]) / 22e-3
    gz_4 = (ht_sensor[3] - ht_sensor[7]) / 22e-3
    
    gz_n_center = np.mean([gz_1, gz_2, gz_3, gz_4]) / ht_center
    
    ht_mid_1 = (ht_sensor[0] + ht_sensor[4]) / 2 
    ht_mid_2 = (ht_sensor[1] + ht_sensor[5]) / 2
    ht_mid_3 = (ht_sensor[2] + ht_sensor[6]) / 2
    ht_mid_4 = (ht_sensor[3] + ht_sensor[7]) / 2
    
    gz_n_1 = gz_1 / ht_mid_1; gz_n_2 = gz_2 / ht_mid_2; gz_n_3 = gz_3 / ht_mid_3; gz_n_4 = gz_4 / ht_mid_4     
    
    ht_tip_1 = ht_mid_1 * extrapolation_factor_fitted(gz_n_1)
    ht_tip_2 = ht_mid_2 * extrapolation_factor_fitted(gz_n_2)
    ht_tip_3 = ht_mid_3 * extrapolation_factor_fitted(gz_n_3)
    ht_tip_4 = ht_mid_4 * extrapolation_factor_fitted(gz_n_4)
    ht_tip_max = np.max([ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4])
    ht_tip_avg = np.mean([ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4])
    ht_tip = [ht_tip_max, ht_tip_avg]
    ht_tip_rms = [x/np.sqrt(2) for x in ht_tip]    
        
    return h_center_rms, gz_n_center, ht_tip_rms
    
# find files whose names meet the specified pattern under the specified directory
def get_files(file_dir, file_pattern):
    full_pattern = os.path.join(file_dir, file_pattern)
    files = glob.glob(full_pattern)
    files.sort()
    
    return files

# extract compressed files whose names start with the specified string to the specified directory
# if out_path does not exist, it will be created first 
def extract_zip_files(zip_files, file_header, out_dir):
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file) as archive:
            for file in archive.namelist():
                if file.startswith(file_header):
                    archive.extract(file, out_dir)

# derive the envelope of the time-domain signal with moving averaging, from Shihao
def compute_moving_average(input_data, window_size = 12000):
    # compute mvg output
    input_data = np.abs(input_data)
    n = window_size
    ret = np.cumsum(input_data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    output = ret[n - 1:] / n # len(output) = len(input_data) - window_size + 1

    # add tail and head constant value to make the output of the same length as the input
    tail = np.repeat(output[-1], int(window_size / 2))
    head = np.repeat(output[0], int(window_size / 2)-1)
    output = np.concatenate((head, output))
    output = np.concatenate((output, tail))
    return output

# calculate the minimum size of the slice of the time-domain signal, from Shihao
# default min_cycles set to 5, corresponding to the implementation of MAGPy handheld V2.0 and Module WPT V2.0
def compute_minimal_slice_window_size(user_defined_frequency, min_cycles = 5):
    computed_min_slice_window_size = int(np.ceil(min_cycles * (1 / user_defined_frequency) * 25000000))
    print("computed_min_slice_window_size: ", computed_min_slice_window_size)
    return computed_min_slice_window_size

# determine the start and stop index of the slice of the time-domain signal, from Shihao
def compute_slice_indices(input_data, decay_threshold = 0.95, is_peak_frame = True, min_window_size = 1024):
    min_slice_window_size = max(min_window_size, 1024) # 1024 corresponding to the implementation of MAGPy handheld V2.0 and Module WPT V2.0

    print("final decided min_slice_window_size: ", min_slice_window_size)

    input_size = input_data.shape[0]
    mid_pos = int(input_size / 2)

    if(not is_peak_frame):
        mid_pos = np.argmax(input_data)

    print("Mid position of slice window: ", mid_pos)

    mid_value = input_data[mid_pos]

    threshold = mid_value * decay_threshold

    left_pos = mid_pos
    right_pos = mid_pos

    left_count = 1
    right_count = 1

    while(left_pos > 0 and input_data[left_pos] >= threshold):
        left_pos -= 1
        left_count += 1

    while(right_pos < (input_size - 1) and input_data[right_pos] >= threshold):
        right_pos += 1
        right_count += 1

    # when the slice window size is smaller than the minimal window size
    while( (left_count + right_count - 1) < min_slice_window_size ):
        if left_pos > 0:
            left_pos -= 1
            left_count += 1
        if right_pos < (input_size - 1):
            right_pos += 1
            right_count += 1
    return left_pos, right_pos


def generate_pulse_signal(f_s=25e6, duration=6e-3, f_c=100e3, phase_shift=0, f_m=1e2, mod_index=1, envelope_shape='square', duty_cycle=0.5, ramp_applied=True, ramp_time_rel=0.2, noise_added=False, snr_db=30):
    '''    
    ---global param
    f_s: sample rate
    duration: how long the signal is along the time axis
    ----param about the carrier wave
    f_c: freq of the carrier wave    
    phase_shift: initial phase of the carrier wave
    ---param about the modulation
    f_m: freq of the modulation envelope, f_m < f_c
    mod_index: modulation index, = (V_on - V_off)/(V_on + V_off), within (0, 1)
    envelope_shape: square or triangle or sine
    duty_cycle: only appliable to square-wave envelope
    ---param about ramp up/down
    ramp_applied: enable the Gaussian ramp up/down
    ramp_time_rel: ramp-up/down time, defined in terms of half the period of the envelope
    ----param about noise
    noise_added: enable adding the random noise
    snr_db: signal-to-noise ratio in dB, = 20*log10(1 / stdev of noise)
    ----notes
    1. the pulse signal has a unit amplitude
    '''
    t = np.linspace(0, duration, int(f_s*duration), endpoint=False)
    carrier = np.sin(2*np.pi*f_c*t + phase_shift)
    if envelope_shape == 'square':
        envelope = np.abs((1/(mod_index+1)) * (signal.square(2*np.pi*f_m*t, duty=duty_cycle) + mod_index))
        if ramp_applied == True:
            ramp_up_index = np.insert(np.where( np.isclose(np.diff(envelope), (1-(1-mod_index)/(mod_index+1))) )[0] + 1, 0, 0)
            ramp_down_index = np.where( np.isclose(np.diff(envelope), ((1-mod_index)/(mod_index+1)-1)) )[0]
            ramp_samples = int(ramp_time_rel*(0.5/f_m)*f_s)            
            window = signal.windows.gaussian(ramp_samples*2, std=ramp_samples/3)
            for i in ramp_up_index:
                envelope[i:i+ramp_samples] = (1-(1-mod_index)/(mod_index+1))*envelope[i:i+ramp_samples]*window[0:ramp_samples] + (1-mod_index)/(mod_index+1)
            for i in ramp_down_index:
                envelope[i+1-ramp_samples:i+1] = (1-(1-mod_index)/(mod_index+1))*envelope[i+1-ramp_samples:i+1]*window[-ramp_samples:] + (1-mod_index)/(mod_index+1)         
    elif envelope_shape == 'triangle':
        envelope = np.abs((1/(mod_index+1)) * (signal.sawtooth(2*np.pi*f_m*t, width=0.5) + mod_index))
    else:
        envelope = np.abs((1/(mod_index+1)) * (np.sin(2*np.pi*f_m*t) + mod_index))        
    
    sig = carrier*envelope
    
    if noise_added == True:
        one_sigma = 10**(-1*snr_db/20)
        noise = np.clip(one_sigma*np.random.randn(len(t)), -3.0*one_sigma, 3.0*one_sigma)  # 1-sigma definition used, 3-sigma clamp applied
        sig = sig + noise
    
    return t, envelope, sig  


def calc_induced_efield_from_incident_efield(einc, f):
    eind = 3.79 * sc.epsilon_0 / np.abs(sc.epsilon_0*(1+55) - 1j*0.75/(2*np.pi*f)) * einc
    return eind
