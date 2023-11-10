# safety limits from ICNIRP 1998
# source: Table 4
# applicable frequencies: >0 Hz - 10 MHz
# applicable tissues: originally all tissues of head and trunk, later reduced to CNS tissues of head and trunk (i.e., brain and spinal cord) 
def icnirp1998_jind(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f > 0 and f <= 1:
            jind = 8
        elif f > 1 and f <= 4:
            jind = 8 / f
        elif f > 4 and f <= 1e3:
            jind = 2
        elif f > 1e3 and f <= 10e6:
            jind = f / 500
        else:
            jind = float('nan')
    else:
        if f > 0 and f <= 1:
            jind = 40
        elif f > 1 and f <= 4:
            jind = 40 / f
        elif f > 4 and f <= 1e3:
            jind = 10
        elif f > 1e3 and f <= 10e6:
            jind = f / 100
        else:
            jind = float('nan')
    
    jind = jind*1e-3 # mA/m2 --> A/m2
            
    return jind

def icnirp1998_jind_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0, 1, 4, 1e3, 10e6]
        jind = np.array([8, 8, 2, 2, 2e4])*1e-3
    else:
        f = [0, 1, 4, 1e3, 10e6]
        jind = np.array([40, 40, 10, 10, 1e5])*1e-3
    
    return f, jind

# Eind derived from Jind
# conductivity of nerve (0.348 S/m from IT'IS LF tissue database V4.1) used
def icnirp1998_eind_derived_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0, 1, 4, 1e3, 10e6]
        eind = np.array([8, 8, 2, 2, 2e4])*1e-3/0.348  
    else:
        f = [0, 1, 4, 1e3, 10e6]
        eind = np.array([40, 40, 10, 10, 1e5])*1e-3/0.348
    
    return f, eind

# ICNIRP 1998 does NOT provide separate RL for electical stimulation and thermal effect.

# source: Tables 6 (occupational) and 7 (general public)
# applicable frequencies: >0 Hz (Hinc)/1 Hz (Einc) - 300 GHz, so further implementation needed here
def icnirp1998_hinc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f > 0 and f <=1:
            hinc = 3.2e4
        elif f > 1 and f <= 8:
            hinc = 3.2e4 / (f**2)
        elif f > 8 and f <= 0.8e3:
            hinc = 4e3 / f
        elif f > 0.8e3 and f <= 150e3:
            hinc = 5
        elif f > 0.15e6 and f <= 10e6:
            hinc = 0.73 / (f*1e-6)
        else:
            hinc = float('nan')
    else:
        if f > 0 and f <= 1:
            hinc = 1.63e5
        elif f > 1 and f < 8:
            hinc = 1.63e5 / (f**2)
        elif f >= 8 and f < 0.82e3:
            hinc = 2e4 / f
        elif f >= 0.82e3 and f <= 65e3:
            hinc = 24.4
        elif f > 65e3 and f <=10e6:
            hinc = 1.6 / (f*1e-6)
        else:
            hinc = float('nan')
    
    return hinc

def icnirp1998_einc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 1:
            einc = float('nan')
        elif f >= 1 and f <= 25:
            einc = 1e4
        elif f > 25 and f <3e3:
            einc = 250 / (f*1e-3)
        elif f >= 3e3 and f <= 1e6:
            einc  = 87
        elif f > 1e6 and f <=10e6:
            einc = 87 / (np.sqrt(f*1e-6))
        else:
            einc = float('nan')
    else:
        if f < 1:
            einc = float('nan')
        elif f >= 1 and f <= 25:
            einc = 2e4
        elif f > 25 and f < 0.82e3:
            einc = 500 / (f*1e-3)
        elif f >= 0.82e3 and f <= 1e6:
            einc = 610
        elif f > 1e6 and f <= 10e6:
            einc = 610 / (f*1e-6)
        else:
            einc = float('nan')
    
    return einc

def icnirp1998_hinc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0, 1, 8, 25, 0.8e3, 150e3, 10e6]
        hinc = [3.2e4, 3.2e4, 500, 160, 5, 5, 0.073]            
    else:
        f = [0, 1, 8, 25, 0.82e3, 65e3, 10e6]
        hinc = [1.63e5, 1.63e5, 2.5e3, 800, 24.4, 24.4, 0.16]
    
    return f, hinc

def icnirp1998_einc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [1, 25, 3e3, 1e6, 10e6]
        einc = [1e4, 1e4, 87, 87, 28]            
    else:
        f = [1, 25, 0.82e3, 1e6, 10e6]
        einc = [2e4, 2e4, 610, 610, 61]
    
    return f, einc


# safety limits from ICNIRP 2010
# source: Table 2
# applicable frequencies: 1 Hz - 10 MHz
# applicable tissues: CNS tissue of the head, all tissues of head and body
# note: limits for the two groups of tissues only differ below 1 kHz (general public) or 0.4 kHz (occupational)
def icnirp2010_eind(f, exposure_scenario='public', body_part='head_and_body'):
    if exposure_scenario == 'public':
        if body_part == 'head_cns':
            if f >= 1 and f < 10:
                eind = 0.1 / f
            elif f >= 10 and f <= 25:
                eind = 0.01
            elif f > 25 and f < 1e3:
                eind = 4e-4 * f
            elif f >= 1e3 and f <= 3e3:
                eind = 0.4
            elif f > 3e3 and f <= 10e6:
                eind = 1.35e-4 * f
            else:
                eind = float('nan')        
        elif body_part == 'head_and_body':
            if f >=1 and f <= 3e3:
                eind = 0.4
            elif f > 3e3 and f <=10e6:
                eind = 1.35e-4 * f
            else: 
                eind = float('nan')        
        else:
            eind = float('nan')        
    else:
        if body_part == 'head_cns':
            if f >= 1 and f < 10:
                eind = 0.5 / f
            elif f >= 10 and f <= 25:
                eind = 0.05
            elif f > 25 and f < 400:
                eind = 2e-3 * f
            elif f >= 400 and f <= 3e3:
                eind = 0.8
            elif f > 3e3 and f <= 10e6:
                eind = 2.7e-4 * f
            else:
                eind = float('nan')
        elif body_part == 'head_and_body':
            if f >= 1 and f <= 3e3:
                eind = 0.8
            elif f > 3e3 and f <= 10e6:
                eind = 2.7e-4 * f
            else:
                eind = float('nan')
        else:
            eind = float('nan')    
    
    return eind

def icnirp2010_eind_trace(exposure_scenario='public', body_part='head_and_body'):
    if exposure_scenario == 'public':
        if body_part == 'head_cns':
            f = [1, 10, 25, 1e3, 3e3, 10e6]
            eind = [0.1, 0.01, 0.01, 0.4, 0.4, 1.35e3]
        elif body_part == 'head_and_body':
            f = [1, 3e3, 10e6]
            eind = [0.4, 0.4, 1.35e3]
        else:
            f = float('nan')
            eind = float('nan')
    else:
        if body_part == 'head_cns':
            f = [1, 10, 25, 400, 3e3, 10e6]
            eind = [0.5, 0.05, 0.05, 0.8, 0.8, 2.7e3]
        elif body_part == 'head_and_body':
            f = [1, 3e3, 10e6]
            eind = [0.8, 0.8, 2.7e3]
        else:
            f = float('nan')
            eind = float('nan')
    
    return f, eind

# ICNIRP 2010 only provides RL for electrical stimulation.

# source: Tables 3 (occupational) and 4 (general public)
# applicable freqeuncies: 1 Hz - 10 MHz
def icnirp2010_hinc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 1 and f <=8:
            hinc = 3.2e4 / f**2
        elif f > 8 and f < 25:
            hinc = 4e3 / f
        elif f >= 25 and f <= 400:
            hinc = 1.6e2
        elif f> 400 and f < 3e3:
            hinc = 6.4e4 / f
        elif f >= 3e3 and f <= 10e6:
            hinc = 21
        else:
            hinc = float('nan')
    else:
        if f >= 1 and f <= 8:
            hinc = 1.63e5 / f**2
        elif f > 8 and f < 25:
            hinc = 2e4 / f
        elif f >= 25 and f <= 300:
            hinc = 8e2
        elif f > 300 and f < 3e3:
            hinc = 2.4e5 / f
        elif f >= 3e3 and f <= 10e6:
            hinc = 80
        else:
            hinc = float('nan')
    
    return hinc

def icnirp2010_einc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 1 and f <= 50:
            einc = 5 * 1e3
        elif f > 50 and f < 3e3:
            einc = 2.5e2 / f * 1e3
        elif f >= 3e3 and f <= 10e6:
            einc = 8.3e-2 * 1e3
        else:
            einc = float('nan')
    else:
        if f >=1 and f <= 25:
            einc = 20 * 1e3
        elif f > 25 and f < 3e3:
            einc = 5e2 / f * 1e3
        elif f >= 3e3 and f <= 10e6:
            einc = 1.7e-1 * 1e3
        else:            
            einc = float('nan')
    
    return einc

def icnirp2010_hinc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [1, 8, 25, 400, 3e3, 10e6]
        hinc = [3.2e4, 5e2, 1.6e2, 1.6e2, 21, 21]
    else:
        f = [1, 8, 25, 300, 3e3, 10e6]
        hinc = [1.63e5, 2.5e3, 8e2, 8e2, 80, 80]
    
    return f, hinc

def icnirp2010_einc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [1, 50, 3e3, 10e6]
        einc = [5e3, 5e3, 83, 83]
    else:
        f = [1, 25, 3e3, 10e6]
        einc = [20e3, 20e3, 1.7e2, 1.7e2]
    
    return f, einc 

# safety limits from ICNIRP 2020
# source: Table 4
# applicable frequencies: 100 kHz - 10 MHz 
# applicable tissues: all tissues of head and body
def icnirp2020_eind(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 100e3 and f <= 10e6:
            eind = 1.35e-4 * f
        else:
            eind = float('nan')
    else:
        if f >= 100e3 and f <=10e6:
            eind = 2.70e-4 * f
        else:
            eind = float('nan')
    
    return eind

def icnirp2020_eind_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 10e6]
        eind = 1.35e-4 * np.array(f)
    else:
        f = [100e3, 10e6]
        eind = 2.70e-4 * np.array(f)
    
    return f, eind

# Eind derived from SAR for head and trunk
# mass density (1000 kg/m3) and conductivity of muscle (0.461 S/m from IT'IS LF tissue database V4.1) used
def icnirp2020_eind_derived_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 10e6]
        eind = [np.sqrt(2*1e3/0.461), np.sqrt(2*1e3/0.461)]
    else:
        f = [100e3, 10e6]
        eind = [np.sqrt(10*1e3/0.461), np.sqrt(10*1e3/0.461)]
    
    return f, eind

# Eind derived from SAR for limbs
# mass density (1000 kg/m3) and conductivity of skin (0.148 S/m from IT'IS LF tissue database V4.1) used
def icnirp2020_eind_derived_trace_2(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 10e6]
        eind = [np.sqrt(4*1e3/0.148), np.sqrt(4*1e3/0.148)]
    else:
        f = [100e3, 10e6]
        eind = [np.sqrt(20*1e3/0.148), np.sqrt(20*1e3/0.148)]
    
    return f, eind

# ICNIRP 2020 provides separate RL for electrical stimulation and thermal effect.

# source: Table 8 (local exposure)
# applicable frequencies: 100 kHz - 10 MHz
def icnirp2020_hinc_stimulation(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 100e3 and f <= 10e6:
            hinc = 21
        else:
            hinc = float('nan')
    else:
        if f >= 100e3 and f <= 10e6:
            hinc = 80
        else:
            hinc = float('nan')
        
    return hinc

def icnirp2020_einc_stimulation(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 100e3 and f <= 10e6:
            einc = 83
        else:
            einc = float('nan')
    else:
        if f >= 100e3 and f <= 10e6:
            einc = 170
        else:
            einc = float('nan')
        
    return einc

# source: Table 6 (local exposure)
# applicable frequencies: 100 kHz - 2 GHz
def icnirp2020_hinc_thermal(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 100e3:
            hinc = float('nan')
        elif f >= 100e3 and f <= 30e6:
            hinc = 4.9 / (f*1e-6)
        else:
            hinc = float('nan')
    else:
        if f < 100e3:
            hinc = float('nan')
        elif f >= 100e3 and f <= 30e6:
            hinc = 10.8 / (f*1e-6)
        else:
            hinc = float('nan')
    
    return hinc

def icnirp2020_einc_thermal(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 100e3:
            einc = float('nan')
        elif f >= 100e3 and f <= 30e6:
            einc = 671 / (f*1e-6)**0.7
        else:
            einc = float('nan')
    else:
        if f < 100e3:
            einc = float('nan')
        elif f >= 100e3 and f <= 30e6:
            einc = 1504 / (f*1e-6)**0.7
        else:
            einc = float('nan')
    
    return einc

def icnirp2020_hinc_stimulation_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 10e6]
        hinc = [21, 21]
    else:
        f = [100e3, 10e6]
        hinc = [80, 80]
        
    return f, hinc

def icnirp2020_einc_stimulation_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 10e6]
        einc = [83, 83]
    else:
        f = [100e3, 10e6]
        einc = [170, 170]
        
    return f, einc

def icnirp2020_hinc_thermal_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 30e6]
        hinc = [4.9/(100e3*1e-6), 4.9/(30e6*1e-6)]
    else:
        f = [100e3, 30e6]
        hinc = [10.8/(100e3*1e-6), 10.8/(30e6*1e-6)]
    
    return f, hinc

def icnirp2020_einc_thermal_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [100e3, 30e6]
        einc = [671/(100e3*1e-6)**0.7, 671/(30e6*1e-6)**0.7]
    else:
        f = [100e3, 30e6]
        einc = [1504/(100e3*1e-6)**0.7, 1504/(30e6*1e-6)**0.7]
    
    return f, einc

# safety limits from IEEE 2019
# source: table 1
# applicable frequencies: 0 Hz - 5 MHz
def ieee2019_eind(f, exposure_scenario='public', body_part='brain'):
    if exposure_scenario == 'public':
        if body_part == 'brain':
            if f < 20:
                eind = 5.89e-3
            elif f <= 5e6:
                eind = 5.89e-3 * (f/20)
            else:
                eind = float('nan')
        elif body_part == 'heart':
            if f < 167:
                eind = 0.943
            elif f <= 5e6:
                eind = 0.943 * (f/167)
            else:
                eind = float('nan')
        elif body_part == 'limbs':
            if f < 3350:
                eind = 2.1
            elif f <= 5e6:
                eind = 2.1 * (f/3350)
            else:
                eind = float('nan')
        else:
            if f < 3350:
                eind = 0.701
            elif f <= 5e6:
                eind = 0.701 * (f/3350)
            else:
                eind = float('nan')
    else:
        if body_part == 'brain':
            if f < 20:
                eind = 1.77e-2
            elif f <= 5e6:
                eind = 1.77e-2 * (f/20)
            else:
                eind = float('nan')
        elif body_part == 'heart':
            if f < 167:
                eind = 0.943
            elif f <= 5e6:
                eind = 0.943 * (f/167)
            else:
                eind = float('nan')
        elif body_part == 'limbs':
            if f < 3350:
                eind = 2.1
            elif f <= 5e6:
                eind = 2.1 * (f/3350)
            else:
                eind = float('nan')
        else:
            if f < 3350:
                eind = 2.1
            elif f <= 5e6:
                eind = 2.1 * (f/3350)
            else:
                eind = float('nan')
        
    return eind    

def ieee2019_eind_trace(exposure_scenario='public', body_part='brain'):
    if exposure_scenario == 'public':
        if body_part == 'brain':
            f = [0, 1, 20, 5e6] # 1 Hz added to facilitate drawing loglog plot
            eind = [5.89e-3, 5.89e-3, 5.89e-3, 5.89e-3*(5e6/20)]
        elif body_part == 'heart':
            f = [0, 1, 167, 5e6]
            eind = [0.943, 0.943, 0.943, 0.943*(5e6/167)]
        elif body_part == 'limbs':
            f = [0, 1, 3350, 5e6]
            eind = [2.1, 2.1, 2.1, 2.1*(5e6/3350)]
        else:
            f = [0, 1, 3350, 5e6]
            eind = [0.701, 0.701, 0.701, 0.701*(5e6/3350)]
    else:
        if body_part == 'brain':
            f = [0, 1, 20, 5e6]
            eind = [1.77e-2, 1.77e-2, 1.77e-2, 1.77e-2*(5e6/20)]
        elif body_part == 'heart':
            f = [0, 1, 167, 5e6]
            eind = [0.943, 0.943, 0.943, 0.943*(5e6/167)]
        elif body_part == 'limbs':
            f = [0, 1, 3350, 5e6]
            eind = [2.1, 2.1, 2.1, 2.1*(5e6/3350)]
        else:
            f = [0, 1, 3350, 5e6]
            eind = [2.1, 2.1, 2.1, 2.1*(5e6/3350)]
    
    return f, eind           

# source: tables 2 and 3
def ieee2019_hinc_stimulation(f, exposure_scenario='public', body_part='head_and_trunk'):
    if exposure_scenario == 'public':
        if body_part == 'head_and_trunk':
            if f <= 0.153:
                hinc = 9.39e4
            elif f < 20:
                hinc = 1.44e4 / f
            elif f <=751:
                hinc = 719
            elif f < 3.35e3:
                hinc = 5.47e5 / f
            elif f <= 5e6:
                hinc = 163
            else:
                hinc = float('nan')
        else: # for limbs
            if f <= 10.7:
                hinc = 2.81e5            
            elif f < 3.35e3:
                hinc = 3.02e6 / f
            elif f <= 5e6:
                hinc = 900
            else:
                hinc = float('nan')
    else:
        if body_part == 'head_and_trunk':
            if f <= 0.153:
                hinc = 2.81e5
            elif f < 20:
                hinc = 4.32e4 / f
            elif f <=751:
                hinc = 2.16e3
            elif f < 3.35e3:
                hinc = 1.64e6 / f
            elif f <= 5e6:
                hinc = 490
            else:
                hinc = float('nan')
        else: # for limbs
            if f <= 10.7:
                hinc = 2.81e5            
            elif f < 3.35e3:
                hinc = 3.02e6 / f
            elif f <= 5e6:
                hinc = 900
            else:
                hinc = float('nan')
                
    return hinc

# source: table 4
def ieee2019_einc_stimulation(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f <= 368:
            einc = 5e3
        elif f < 3e3:
            einc = 1.84e6 / f
        elif f <= 100e3:
            einc = 614
        else:
            einc = float('nan')
    else:
        if f <= 272:
            einc = 2e4
        elif f < 2953:
            einc = 5.44e6 / f
        elif f <= 100e3:
            einc = 1842
        else:
            einc = float('nan')
    
    return einc
        
    
# source: tables 9 and 10
def ieee2019_hinc_thermal(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 0.1e6 and f <= 30e6:
            hinc = 36.4 / (f*1e-6)
        else:
            hinc = float('nan')
    else:
        if f >= 0.1e6 and f <= 100e6:
            hinc = 36.4 / (f*1e-6)
        else:
            hinc = float('nan')
    
    return hinc

def ieee2019_einc_thermal(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 0.1e6:
            einc = float('nan')
        elif f <= 1.34e6:
            einc = 1373
        elif f < 30e6:
            einc = 1842 / (f*1e-6)
        elif f <= 100e6:
            einc = 61.4
        else:
            einc = float('nan')
    else:
        if f < 0.1e6:
            einc = float('nan')
        elif f <= 1e6:
            einc = 4119
        elif f < 30e6:
            einc = 4119 / (f*1e-6)
        elif f <= 100e6:
            einc = 137.3
        else:
            einc = float('nan')
    
    return einc

def ieee2019_hinc_stimulation_trace(exposure_scenario='public', body_part='head_and_trunk'):
    if exposure_scenario == 'public':
        if body_part == 'head_and_trunk':
            f = [0, 0.153, 20, 751, 3.35e3, 5e6]
            hinc = [9.39e4, 9.39e4, 719, 719, 163, 163]
        else:
            f = [0, 1, 10.7, 3350, 5e6]  # 1 Hz added to facilitate drawing loglog plot
            hinc = [2.81e5, 2.81e5, 2.81e5, 900, 900]
    else:
        if body_part == 'head_and_trunk':
            f = [0, 0.153, 20, 751, 3.35e3, 5e6]
            hinc = [2.81e5, 2.81e5, 2.16e3, 2.16e3, 490, 490]
        else:
            f = [0, 1, 10.7, 3350, 5e6] # 1 Hz added to facilitate drawing loglog plot
            hinc = [2.81e5, 2.81e5, 2.81e5, 900, 900]
    
    return f, hinc

def ieee2019_einc_stimulation_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0, 1, 368, 3e3, 100e3] # 1 Hz added to facilitate drawing loglog plot
        einc = [5e3, 5e3, 5e3, 614, 614]
    else:
        f = [0, 1, 272, 2953, 100e3] # 1 Hz added to facilitate drawing loglog plot
        einc = [2e4, 2e4, 2e4, 1842, 1842]
    
    return f, einc
    
def ieee2019_hinc_thermal_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0.1e6, 30e6]
        hinc = [36.4/0.1, 36.4/30]
    else:
        f = [0.1e6, 100e6]
        hinc = [36.4/0.1, 36.4/100]
    
    return f, hinc

def ieee2019_einc_thermal_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0.1e6, 1.34e6, 30e6, 100e6]
        einc = [1373, 1373, 61.4, 61.4]
    else:
        f = [0.1e6, 1e6, 30e6, 100e6]
        einc = [4119, 4119, 137.3, 137.3]
    
    return f, einc

# safety limits from HC Safety Code 6 (2015)
# source: table 1
# applicable frequencies: 3 kHz - 10 MHz
def sc6_eind(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 3e3 and f <= 10e6:
            eind = 1.35e-4 * f
        else:
            eind = float('nan')
    else:
        if f >= 3e3 and f <=10e6:
            eind = 2.70e-4 * f
        else:
            eind = float('nan')
    
    return eind

def sc6_eind_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [3e3, 10e6]
        eind = [0.405, 1350]
    else:
        f = [3e3, 10e6]
        eind = [0.81, 2700]
    
    return f, eind

# HC SC6 2015 provides separate RL for electrical stimulation and thermal effect.

# source: table 4
def sc6_hinc_stimulation(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 3e3 and f <= 10e6:
            hinc = 90
        else:
            hinc = float('nan')
    else:
        if f >= 3e3 and f <= 10e6:
            hinc = 180
        else:
            hinc = float('nan')
    
    return hinc

# source: table 3
def sc6_einc_stimulation(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 3e3 and f <= 10e6:
            einc = 83
        else:
            einc = float('nan')
    else:
        if f >= 100e3 and f <= 10e6:
            einc = 170
        else:
            einc = float('nan')
    return einc

# source: table 4
def sc6_hinc_thermal(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 100e3 and f <= 10e6:
            hinc = 0.73 / (f*1e-6)
        else:
            hinc = float('nan')
    else:
        if f >= 100e3 and f <= 10e6:
            hinc = 1.6 / (f*1e-6)
        else:
            hinc = float('nan')
    return hinc

# source: table 3
def sc6_einc_thermal(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f >= 100e3 and f <= 10e6:
            einc = 87 / (f*1e-6)**0.5
        else:
            einc = float('nan')
    else:
        if f >= 100e3 and f <= 10e6:
            einc = 193 / (f*1e-6)**0.5
        else:
            einc = float('nan')
    return einc

def sc6_hinc_stimulation_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [3e3, 10e6]
        hinc = [90, 90]
    else:
        f = [3e3, 10e6]
        hinc = [180, 180]
    
    return f, hinc

def sc6_einc_stimulation_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [3e3, 10e6]
        einc = [83, 83]
    else:
        f = [3e3, 10e6]
        einc = [170, 170]
    
    return f, einc

def sc6_hinc_thermal_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [0.1e6, 10e6]
        hinc = [7.3, 0.073]
    else:
        f = [0.1e6, 10e6]
        hinc = [16, 0.16]
    
    return f, hinc

def sc6_einc_thermal_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [1e6, 10e6]
        einc = [87, 28]
    else:
        f = [1e6, 10e6]
        einc = [193, 61]
    
    return f, einc

# safety limits from FCC
# Eind limits described in Table 2 of FCC Docket 19-226 released on Apr. 6, 2020 are 
# the same as HC SC6-2015, so no additional implementation was made here.

# source 1: table 1 of FCC Docket 19-126 released on Apr. 6, 2020 (valid down to 300 kHz)
# source 2: KDB 680106, released on Oct. 24, 2023 (providing RL below 300 kHz, for both general public and occupational exposures?)
def fcc_hinc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 3e3: # assume the limit valid down to 3 kHz
            hinc = float('nan')
        elif f < 100e3:
            hinc = 90
        elif f <= 1.34e6:
            hinc = 1.63 # 100 kHz should use this value
        elif f <= 30e6:
            hinc = 2.19 / (f*1e-6) # valid up to 30 MHz
        else:
            hinc = float('nan')
    else:
        if f < 3e3:
            hinc = float('nan')
        elif f < 100e3:
            hinc = 90 # assume the same limit as general public exposure
        elif f <= 3e6:
            hinc = 1.63 # assume the same limit as general public exposure
        elif f <= 30e6:
            hinc = 4.89 / (f*1e-6) # valid up to 30 MHz
        else:
            hinc = float('nan')
    
    return hinc

def fcc_einc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 3e3: # assume the limit valid down to 3 kHz
            einc = float('nan')
        elif f < 100e3:
            einc = 83
        elif f <= 1.34e6:
            einc = 614 # 100 kHz should use this value
        elif f <= 30e6:
            einc = 824 / (f*1e-6) # valid up to 30 MHz
        else:
            einc = float('nan')
    else:
        if f < 3e3:
            einc = float('nan')
        elif f < 100e3:
            einc = 83 # assume the same limit as general public exposure
        elif f < 3e6:
            einc = 614 # assume the same limit as general public exposure
        elif f < 30e6:
            einc = 1842 / (f*1e-6) # valid up to 30 MHz
        else:
            einc = float('nan')
    
    return einc

def fcc_hinc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [3e3, 100e3, 100e3, 1.34e6, 30e6]
        hinc = [90, 90, 1.63, 1.63, 2.19/30]
    else:
        f = [3e3, 100e3, 100e3, 3e6, 30e6]
        hinc = [90, 90, 1.63, 1.63, 4.89/30] # limits below 300 kHz assumed to the same as general public exposure
    
    return f, hinc

def fcc_einc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [3e3, 100e3, 100e3, 1.34e6, 30e6]
        einc = [83, 83, 614, 614, 824/30]
    else:
        f = [3e3, 100e3, 100e3, 3e6, 30e6] 
        einc = [83, 83, 614, 614, 1842/30] # limits below 300 kHz assumed to the same as general public exposure
    
    return f, einc

# safety limits from Chinese standard GB 8702-2014
# there is no basic restriction in GB 8702-2014 
# source: table 1 (only for general public exposure scenario)
def gb2014_hinc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 1:
            hinc = float('nan')
        elif f <= 8:
            hinc = 3.2e4 / f**2
        elif f < 1.2e3:
            hinc = 4e3 / f
        elif f <= 2.9e3:
            hinc = 3.3
        elif f < 100e3:
            hinc = 10 / (f*1e-3)
        elif f <= 3e6:
            hinc = 0.1
        elif f <= 30e6:
            hinc = 0.17 / (f*1e-6)**0.5
        else:
            hinc = float('nan')
    else:
        hinc = float('nan')
    
    return hinc

def gb2014_einc(f, exposure_scenario='public'):
    if exposure_scenario == 'public':
        if f < 1:
            einc = float('nan')
        elif f <= 25:
            einc = 8e3
        elif f < 2.9e3:
            einc = 200 / (f*1e-3)
        elif f <= 57e3:
            einc = 70
        elif f < 100e3:
            einc = 4e3 / (f*1e-3)
        elif f <= 3e6:
            einc = 40
        elif f <= 30e6:
            einc = 67 / (f*1e-6)**0.5
        else:
            einc = float('nan')
    else:
        einc = float('nan')
    
    return einc

def gb2014_hinc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [1, 8, 1.2e3, 2.9e3, 100e3, 3e6, 30e6]
        hinc = [3.2e4, 4e3/8, 3.3, 3.3, 0.1, 0.1, 0.17/30**0.5]
    else:
        f = float('nan')
        hinc = float('nan')
    
    return f, hinc

def gb2014_einc_trace(exposure_scenario='public'):
    if exposure_scenario == 'public':
        f = [1, 25, 2.9e3, 57e3, 0.1e6, 30e6]
        einc = [8e3, 8e3, 70, 70, 40, 67/30**0.5]
    else:
        f = float('nan')
        einc = float('nan')        
    
    return f, einc
