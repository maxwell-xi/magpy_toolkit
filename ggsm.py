import numpy as np

def induced_e_cube(g_n, field, e_br, hfield_used=True): # valid for ICNIRP 2010 & ICNIRP 2020
    if g_n < 0:
        print('Negative normalized gradent encountered! Absolute value used to derive the induced field.')
        g_n = np.abs(g_n)
    k = 1/((1+6.5e-6*g_n**5.8)**(1/5.8))
    if hfield_used == True:
        e_est = k * field / 21 * e_br # H0 = 21 A/m
    else:
        e_est = k * field / 27 * e_br  # B0 = 27 uT
    return e_est

def induced_e_line(g_n, field, e_br, hfield_used=True): # valid for IEEE 2005 & IEEE 2019
    if g_n < 0:
        print('Negative normalized gradent encountered! Absolute value used to derive the induced field.')
        g_n = np.abs(g_n)
    k = 1/((1+4e-11*g_n**6.6)**(1/6.6))
    if hfield_used == True:
        e_est = k * field / 163 * e_br # H0 = 163 A/m
    else:
        e_est = k * field / 205 * e_br # B0 = 205 uT
    return e_est
   
def induced_j_area(g_n, field, j_br, hfield_used=True): # valid for ICNIRP 1998
    if g_n < 0:
        print('Negative normalized gradent encountered! Absolute value used to derive the induced field.')
        g_n = np.abs(g_n)
    k = 1/((1+4e-3*g_n**2.9)**(1/2.9))
    if hfield_used == True:
        j_est = k * field / 5 * j_br # H0 = 5 A/m
    else:
        j_est = k * field / 6.25 * j_br # B0 = 6.25 uT
    return j_est
    
def sar_10g(g_n, field, f, hfield_used=True): # valid for ICNIRP 1998/2020 and IEEE 2005/2019
    if g_n < 0:
        print('Negative normalized gradent encountered! Absolute value used to derive the induced field.')
        g_n = np.abs(g_n)
    k = ( 1/((1+2e-1*g_n**1.2)**(1/1.2)) )**2
    if hfield_used == True:
        sar_est = k * (field/5)**2 * (f/100e3)**2 * 2 * 2e-4 # H0 = 5 A/m, SAR_BR = 2 W/kg
    else:
        sar_est = k * (field/6.25)**2 * (f/100e3)**2 * 2 * 2e-4 # B0 = 5 A/m, SAR_BR = 2 W/kg
    return sar_est

def sar_1g(g_n, field, f, hfield_used=True): # valid for FCC and ISED
    if g_n < 0:
        print('Negative normalized gradent encountered! Absolute value used to derive the induced field.')
        g_n = np.abs(g_n)
    k = ( 1/((1+2.5e-1*g_n**1.1)**(1/1.1)) )**2
    if hfield_used == True:
        sar_est = k * (field/1.63)**2 * (f/100e3)**2 * 1.6 * 4.6e-5 # H1 = 1.63 A/m, SAR_BR = 1.6 W/kg
    else:
        sar_est = k * (field/2.04)**2 * (f/100e3)**2 * 1.6 * 4.6e-5 # B1 = 2.04 uT, SAR_BR = 1.6 W/kg        
    return sar_est

def induced_e_local(g_n, field, e_br, hfield_used=True): # valid for ISED
    if g_n < 0:
        print('Negative normalized gradent encountered! Absolute value used to derive the induced field.')
        g_n = np.abs(g_n)
    k = 1/((1+6.5e-6*g_n**5.8)**(1/5.8))
    if hfield_used == True:
        e_est = k * field / 21 * e_br # H0 = 21 A/m
    else:
        e_est = k * field / 27 * e_br  # B0 = 27 uT
    return 2*e_est

# set negative gradients to zero, otherwise the coupling-factor calculation cannot proceed
def negative_gradient_check(g_n):
    neg_amount = 0
    
    for i in range(g_n.shape[0]):
        for j in range(g_n.shape[1]):
            if g_n[i, j] < 0:
                g_n[i, j] = 0
                neg_amount = neg_amount + 1
    
    if neg_amount != 0:
        print('Number of negative-gradient data points: {}'.format(neg_amount))
    
    return g_n