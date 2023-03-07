import numpy as np
import pandas as pd
from data_visualization_and_processing import helper

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
    
def extrapolation_factor_fitted(g_n):
    coeff = np.array([1, 1, -1.04, 11.0, -31.7, 45.9, -32.2, 8.95])
    g_d_product = g_n * 18.5e-3
    item = np.array([1, g_d_product, g_d_product**2, g_d_product**3, g_d_product**4, g_d_product**5, g_d_product**6, g_d_product**7])
    extrap_factor = np.sum(coeff * item)
    
    return extrap_factor

def mimic_magpy_probe(field, grid_mm, probe_center_loc_mm = [0, 0, 18.5]):    
    i_center = np.argwhere(grid_mm[0] == probe_center_loc_mm[0])[0,0]
    j_center = np.argwhere(grid_mm[1] == probe_center_loc_mm[1])[0,0]
    k_center = np.argwhere(grid_mm[2] == probe_center_loc_mm[2])[0,0]
    ht_center_true = field[3][i_center, j_center, k_center]
    
    k_tip = k_center - 37  # 37 grid lines corresponds to 18.5 mm
    ht_tip_true = field[3][i_center, j_center, k_tip]
    
    i_plus = i_center + 22; i_minus = i_center - 22
    j_plus = j_center + 22; j_minus = j_center - 22
    k_top = k_center + 22; k_bottom = k_center - 22    
  
    ht_tip_true_1 = field[3][i_plus, j_plus, k_tip]
    ht_tip_true_2 = field[3][i_minus, j_plus, k_tip]
    ht_tip_true_3 = field[3][i_minus, j_minus, k_tip]
    ht_tip_true_4 = field[3][i_plus, j_minus, k_tip]
        
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
    ht_tip = np.max([ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4])
    ht_tip_rms = ht_tip/np.sqrt(2)
    
    ht_tip_error_1 = 20*np.log10(ht_tip / np.max([ht_tip_true_1, ht_tip_true_2, ht_tip_true_3, ht_tip_true_4]))
    ht_tip_error_2 = 20*np.log10(ht_tip / ht_tip_true) 
    ht_tip_error = [ht_tip_error_1, ht_tip_error_2]
    
    return h_center_rms, ht_center_error, g_n_center, ht_tip_rms, ht_tip_error

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
    i_center = np.argwhere(grid_mm[0] == probe_center_loc_mm[0])[0,0]
    j_center = np.argwhere(grid_mm[1] == probe_center_loc_mm[1])[0,0]
    k_center = np.argwhere(grid_mm[2] == probe_center_loc_mm[2])[0,0]
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
    ht_tip = np.max([ht_tip_1, ht_tip_2, ht_tip_3, ht_tip_4])
    ht_tip_rms = ht_tip/np.sqrt(2)    
    
    ht_tip_error_1 = 20*np.log10(ht_tip / np.max([ht_tip_true_1, ht_tip_true_2, ht_tip_true_3, ht_tip_true_4]))
    ht_tip_error_2 = 20*np.log10(ht_tip / ht_tip_true) 
    ht_tip_error = [ht_tip_error_1, ht_tip_error_2]    
    
    return h_center_rms, ht_center_error, g_n_center, ht_tip_rms, ht_tip_error  