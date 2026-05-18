import numpy as np
from scipy.special import lpmv  # associated Legendre, order m, degree n

def calc_associated_legendre_func(n_max=20_000):
    # 20k may be not large enough, e.g., for large coils and small distances
    A_cache = np.array([lpmv(1, n, 0.0) for n in range(n_max+1)])  # A_cache[n] = P_n^1(0)
    return A_cache
    

def calc_hfield_r_component_fast(r, theta, a, n_max=20_000):
    """
    Faster version of calc_hfield_r_component.
    Uses:
      - precomputed A_n = P_n^1(0)
      - recurrence for B_n = P_n(cos(theta))
      - iterative power accumulation for r_rel terms
    """
    r_rel = r / a
    # choose n_max as before
    if r_rel < 1 - 1e-3 / a:
        n_max = 200
        regime = "inside_far"
    elif r_rel < 1:
        n_max = n_max
        regime = "inside_near"
    elif np.isclose(r_rel, 1.0):
        # keep your special handling
        print("r = a encountered!")
        return np.nan
    elif r_rel <= 1 + 1e-3 / a:
        n_max = n_max
        regime = "outside_near"
    else:
        n_max = 200
        regime = "outside_far"

    if n_max > len(A_cache) - 1:
        raise ValueError(f"n_max={n_max} exceeds precomputed A_cache size {len(A_cache) - 1}")

    mu = 0  # order for B_n
    x = np.cos(theta)

    # Initialize Legendre P_n(x) via recurrence:
    # P_0(x) = 1, P_1(x) = x
    # (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    P_nm1 = 1.0       # P_0
    P_n   = x         # P_1

    h_sum = 0.0

    # First term: n = 1
    # A_1 = A_cache[1], B_1 = P_1 = x
    if regime in ("inside_far", "inside_near"):
        pow_term = r_rel**0  # r_rel^(n-1) with n=1 → r_rel^0 = 1
    else:
        pow_term = (1.0 / r_rel)**3  # (1/r_rel)^(n+2) with n=1 → (1/r_rel)^3

    for n in range(1, n_max+1):
        # B_n is current P_n
        B_n = P_n

        A_n = A_cache[n]

        # accumulate contribution
        h_sum += A_n * B_n * pow_term

        # update power term
        if regime in ("inside_far", "inside_near"):
            pow_term *= r_rel          # multiply by r_rel for next n
        else:
            pow_term *= (1.0 / r_rel)  # multiply by 1/r_rel for next n

        # update Legendre P_{n+1} for next iteration, unless we are at last n
        if n < n_max:
            # recurrence to get P_{n+1}
            # (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
            P_np1 = ((2*n + 1)*x*P_n - n*P_nm1) / (n + 1)
            P_nm1, P_n = P_n, P_np1

    hfield_r = -1.0 / (2.0 * a) * h_sum
    return hfield_r


def calc_hfield_theta_component_fast(r, theta, a, n_max=20_000):
    r_rel = r / a

    if r_rel < 1 - 1e-3 / a:
        n_max = 200
        regime = "inside_far"
    elif r_rel < 1:
        n_max = n_max
        regime = "inside_near"
    elif np.isclose(r_rel, 1.0):
        print("r = a encountered!")
        return np.nan
    elif r_rel <= 1 + 1e-3 / a:
        n_max = n_max
        regime = "outside_near"
    else:
        n_max = 200
        regime = "outside_far"

    if n_max > len(A_cache) - 1:
        raise ValueError(f"n_max={n_max} exceeds precomputed A_cache size {len(A_cache) - 1}")

    x = np.cos(theta)
    h_sum = 0.0

    # power factor
    if regime in ("inside_far", "inside_near"):
        pow_term = r_rel**0  # for n=1
    else:
        pow_term = (1.0 / r_rel)**3

    # Build P_n^1(x) by recurrence:
    # We will maintain P_nm1=P_{n-1}^1, P_n=P_n^1.
    # Start from n=1:
    P_nm1 = 0.0                                    # P_0^1 = 0
    P_n   = -np.sqrt(max(0.0, 1.0 - x*x))         # P_1^1

    for n in range(1, n_max+1):
        B_n = P_n                                 # current P_n^1(x)
        A_n = A_cache[n]

        if regime in ("inside_far", "inside_near"):
            coef_n = -1.0 / n
        else:
            coef_n = 1.0 / (n + 1)                # - * - in your code

        h_sum += A_n * B_n * coef_n * pow_term

        # update pow_term for next n
        if regime in ("inside_far", "inside_near"):
            pow_term *= r_rel
        else:
            pow_term *= (1.0 / r_rel)

        # recurrence to get P_{n+1}^1
        # (n) P_{n+1}^1 = (2n+1)x P_n^1 - (n+1) P_{n-1}^1 (from general formula)
        P_np1 = ((2*n + 1)*x*P_n - (n + 1)*P_nm1) / n
        P_nm1, P_n = P_n, P_np1

    hfield_theta = -1.0 / (2.0 * a) * h_sum
    return hfield_theta     

def hfield_for_circular_coil_in_spherical_coordinate(dout, din, num_of_turns, r, theta, n_max=20_000):
    if (num_of_turns == 1) or (dout == din):
        a_range = [dout/2]
    else:
        a_range = np.linspace(din/2, dout/2, num_of_turns)
    
    h_r_list = []
    h_theta_list = []
    for a in a_range:     
        h_r_0 = calc_hfield_r_component_fast(r, theta, a, n_max)
        h_theta_0 = calc_hfield_theta_component_fast(r, theta, a, n_max)
        h_r_list.append(h_r_0)
        h_theta_list.append(h_theta_0)
    
    h_r = np.sum(h_r_list)
    h_theta = np.sum(h_theta_list)
    h_tot = np.linalg.norm([h_r, h_theta])
        
    return h_r, h_theta, h_tot  

def hfield_for_circular_coil_in_cartesian_coordinate(dout, din, num_of_turns, x, y, z, n_max=20_000):
    if (num_of_turns == 1) or (dout == din):
        a_range = [dout/2]
    else:
        a_range = np.linspace(din/2, dout/2, num_of_turns)
    
    r = np.linalg.norm([x, y, z])
    theta = np.arctan2(np.linalg.norm([x, y]), z)
    phi = np.arctan2(y, x)
    
    h_x_list = []
    h_y_list = []
    h_z_list = []
    for a in a_range:
        h_r_0 = calc_hfield_r_component_fast(r, theta, a, n_max)
        h_theta_0 = calc_hfield_theta_component_fast(r, theta, a, n_max)
        h_x_0 = (h_r_0*np.sin(theta) - h_theta_0*np.cos(theta)) * np.cos(phi) # use "-" for correct derivation of x component
        h_y_0 = (h_r_0*np.sin(theta) - h_theta_0*np.cos(theta)) * np.sin(phi) # use "-" for correct derivation of y component
        h_z_0 = h_r_0*np.cos(theta) + h_theta_0*np.sin(theta) # use "+" for correct derivation of z component
        h_x_list.append(h_x_0)
        h_y_list.append(h_y_0)
        h_z_list.append(h_z_0)
        
    h_x = np.sum(h_x_list)
    h_y = np.sum(h_y_list)
    h_z = np.sum(h_z_list)
    h_tot = np.linalg.norm([h_x, h_y, h_z])
    
    return h_x, h_y, h_z, h_tot

def fix_nan_value_by_interpolation(x, y):
    y = np.array(y)
    if np.any(np.isnan(y)):
        valid = ~np.isnan(y)
        x_valid = x[valid]
        x_invalid = x[~valid]
        y_valid = y[valid]
        
        # interpolate at all x, then assign only into NaN positions
        y_interp = np.interp(x_invalid, x_valid, y_valid)
        y[~valid] = y_interp
    
    return y

def hfield_for_circular_coil_along_line(dout, din, num_of_turns, x, y, z, n_max=20_000):
    h_x = []
    h_y = []
    h_z = []
    h_tot = []
    
    if isinstance(x, (list, np.ndarray)):    
        for x_0 in x:
            h_x_0, h_y_0, h_z_0, h_tot_0 = hfield_for_circular_coil_in_cartesian_coordinate(dout, din, num_of_turns, x_0, y, z, n_max)
            h_x.append(h_x_0)
            h_y.append(h_y_0)
            h_z.append(h_z_0)
            h_tot.append(h_tot_0)
        
        h_x = fix_nan_value_by_interpolation(x, h_x)
        h_y = fix_nan_value_by_interpolation(x, h_y)
        h_z = fix_nan_value_by_interpolation(x, h_z)
        h_tot = fix_nan_value_by_interpolation(x, h_tot) 
    elif isinstance(y, (list, np.ndarray)): 
        for y_0 in y:
            h_x_0, h_y_0, h_z_0, h_tot_0 = hfield_for_circular_coil_in_cartesian_coordinate(dout, din, num_of_turns, x, y_0, z, n_max)
            h_x.append(h_x_0)
            h_y.append(h_y_0)
            h_z.append(h_z_0)
            h_tot.append(h_tot_0)
            
        h_x = fix_nan_value_by_interpolation(y, h_x)
        h_y = fix_nan_value_by_interpolation(y, h_y)
        h_z = fix_nan_value_by_interpolation(y, h_z)
        h_tot = fix_nan_value_by_interpolation(y, h_tot) 
    elif isinstance(z, (list, np.ndarray)):
        for z_0 in z:
            h_x_0, h_y_0, h_z_0, h_tot_0 = hfield_for_circular_coil_in_cartesian_coordinate(dout, din, num_of_turns, x, y, z_0, n_max)
            h_x.append(h_x_0)
            h_y.append(h_y_0)
            h_z.append(h_z_0)
            h_tot.append(h_tot_0)

        h_x = fix_nan_value_by_interpolation(z, h_x)
        h_y = fix_nan_value_by_interpolation(z, h_y)
        h_z = fix_nan_value_by_interpolation(z, h_z)
        h_tot = fix_nan_value_by_interpolation(z, h_tot)   
    else:
        h_x, h_y, h_z, h_tot = hfield_for_circular_coil_in_cartesian_coordinate(dout, din, num_of_turns, x, y, z, n_max)
    
    return h_x, h_y, h_z, h_tot

def field_decay_extent(r, h, decay_threshold_db=-20):
    if np.diff(r)[0] > 0.1e-3:
        r_interp = np.arange(r[0], r[-1]+0.1e-3, 0.1e-3)
        h_interp = np.interp(r_interp, r, h)
        r = r_interp
        h = h_interp    
    
    h_max = np.max(h)
    h_threshold = 10**(decay_threshold_db/20) * h_max

    if np.min(h) > h_threshold:
        e0 = np.nan
        e1 = np.nan
    else:
        idx = np.argmin(np.abs(h - h_threshold))
        e0 = np.abs(r[idx] - r[0])
        e1 = np.abs(r[idx] - r[np.argmax(h)])
    
    return e0, e1   