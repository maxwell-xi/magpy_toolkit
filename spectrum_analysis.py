import scipy
import numpy as np

# https://ch.mathworks.com/matlabcentral/answers/372516-calculate-windowing-correction-factor
def determine_amplitude_spectrum(signal, sample_rate, window='hann', acf_applied=False):
    if window == 'hann':
        signal_windowed = signal * scipy.signal.windows.hann(len(signal))
        acf = len(signal) / sum(scipy.signal.windows.hann(len(signal))) if acf_applied else 1
    elif window == 'hamming':
        signal_windowed = signal * scipy.signal.windows.hamming(len(signal))
        acf = len(signal) / sum(scipy.signal.windows.hamming(len(signal))) if acf_applied else 1
    elif window == 'blackman':
        signal_windowed = signal * scipy.signal.windows.blackman(len(signal))
        acf = len(signal) / sum(scipy.signal.windows.blackman(len(signal))) if acf_applied else 1
    elif window == 'flattop':
        signal_windowed = signal * scipy.signal.windows.flattop(len(signal))
        acf = len(signal) / sum(scipy.signal.windows.flattop(len(signal))) if acf_applied else 1
    elif window == 'rectangular':
        signal_windowed = signal
        acf = 1
    else:
        print('Undefined window!')
    
    spectrum = acf * 2/len(signal_windowed) * np.abs(np.fft.fft(signal_windowed))[:len(signal_windowed)//2]
    freq = np.fft.fftfreq(len(signal_windowed), 1/sample_rate)[:len(signal_windowed)//2]

    return spectrum, freq


# func from Peter
def determine_amplitude_at_freq_sumsq(signal, freq, sample_rate):
    window = scipy.signal.windows.hamming(len(signal))  
    spectrum = np.abs(np.fft.fft(signal * window))
    bin =  freq / (sample_rate / len(signal)) # index of the bin corresponding to the target freq
    min_bin = max(0, int(bin - 5))
    max_bin = min(len(signal), int(bin + 5))
    energy_gain = np.dot(window, window) / len(window)
    normalization = np.sqrt(2) / np.sqrt(len(signal)) / np.sqrt(energy_gain)
    slice = np.array(spectrum[min_bin:max_bin])
    amplitude = np.sqrt(2*np.dot(normalization*slice, normalization*slice) / len(signal))
    return amplitude


# func from Peter
def determine_amplitude_at_freq_dft(signal, freq, sample_rate):  # bascially the sine/cosine implementation of DFT; fail for the signal from Bosch hob!
    t = np.arange(0, len(signal)) / sample_rate
    test_1 = np.sin(2*np.pi * freq * t)
    test_2 = np.sin(2*np.pi * freq * t + np.pi/2)
    v_1 = np.dot(signal, test_1)
    v_2 = np.dot(signal, test_2)
    amplitude = 2/len(signal) * np.hypot(v_1, v_2) 
    return amplitude


# func from Mischa
def determine_amplitude_at_freq_dft_2(signal, freq, sample_rate):
    correlation_vector = np.arange(0, len(signal))
    correlation_vector = signal*np.exp(correlation_vector*-1j*2*np.pi/sample_rate*freq)
    amplitude =  2/len(signal) * np.abs(np.sum(correlation_vector)) 
    return amplitude


# Note: zero padding will raise the noise floor, so usually is only used to detect the dominant frequency
def determine_peak_freq(signal, sample_rate, padding=5, interpolated=True):
    signal = np.array(signal - np.mean(signal)) # remove DC
    
    sample_num = len(signal)
    sample_num_padded = sample_num*(1+padding)
    bin_size_padded = sample_rate/sample_num_padded
        
    signal_padded = np.zeros(sample_num_padded)
    signal_padded[:sample_num] = signal # signal padded = original signal + trailing zeros
    
    spectrum_padded = 2*(1+padding)/sample_num_padded * np.abs(np.fft.fft(signal_padded))[:sample_num_padded//2]
    freq_padded = np.fft.fftfreq(len(signal_padded), 1/sample_rate)[:sample_num_padded//2]
    #freq_padded = np.linspace(0, sample_rate/2, len(signal_padded)//2, endpoint=False) # equivalent method to define the frequencies (note that endpoint has to be set to False)

    #extract peak by quadratic interpolation
    peak_index = np.argmax(spectrum_padded) # derive peak freq based on the max amplitude

    if interpolated == True:
        y0 = spectrum_padded[peak_index]
        ym1 = spectrum_padded[peak_index-1]
        yp1 = spectrum_padded[peak_index+1]
        peak_loc, _, _ = quadratic_interpolation(ym1, y0, yp1)
        peak_freq = freq_padded[peak_index] + bin_size_padded*(peak_loc) # derive precise peak freq
    else:
        peak_freq = freq_padded[peak_index]

    return peak_freq, spectrum_padded, freq_padded


def quadratic_interpolation(ym1, y0, yp1):
    #QINT - quadratic interpolation of three adjacent samples
    #
    # [p,y,a] = qint(ym1,y0,yp1)
    #
    # returns the extremum location p, height y, and half-curvature a
    # of a parabolic fit through three points.
    # Parabola is given by y(x) = a*(x-p)^2+b,
    # where y(-1)=ym1, y(0)=y0, y(1)=yp1.

    p = (yp1 - ym1) / (2*(2*y0 - yp1 - ym1));
    y = y0 - 0.25*(ym1-yp1)*p;
    a = 0.5*(ym1 - 2*y0 + yp1);
    return p, y, a  


def determine_complex_spectrum(signal, sample_rate):
    spectrum = 2/len(signal) * np.fft.fft(signal)[:len(signal)//2] 
    freq = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(signal)//2]
    spectrum_amplitude = np.abs(spectrum)
    spectrum_angle = np.angle(spectrum, deg=True) # return angles in (-180 deg, 180 deg]
    return spectrum_amplitude, spectrum_angle, freq


# extended version of determine_amplitude_at_freq_dft_2()
def determine_complex_spectrum_at_freq(signal, freq, sample_rate):
    correlation_vector = np.arange(0, len(signal))
    correlation_vector = signal*np.exp(correlation_vector*-1j*2*np.pi/sample_rate*freq)
    complex_spectrum =  2/len(signal) * np.sum(correlation_vector) 
    amplitude = np.abs(complex_spectrum)
    angle_deg = np.angle(complex_spectrum, deg=True)
    return amplitude, angle_deg