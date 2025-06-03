import numpy as np
import scipy.signal as sg
import scipy.constants as const
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.special import erf
import matplotlib.pyplot as plt

# -------------------------------- Machine Constants  ---------------------------------- #
# ---- HSR Frequencies ---- #
HSR_P_REV_FREQ_STORE    = 78194.34      # - Hz
HSR_P_REV_FREQ_INJ      = 78133.5651    # - Hz
HSR_Au_REV_FREQ_STORE   = 78194.34      # - Hz
HSR_Au_REV_FREQ_INJ     = 77840.6       # - Hz

HSR_RF_HARMONIC_NUMBER  = 315
HSR_RF_24M  = HSR_RF_HARMONIC_NUMBER * HSR_P_REV_FREQ_STORE
HSR_RF_49M  = 2 * HSR_RF_24M
HSR_RF_98M  = 2 * HSR_RF_49M
HSR_RF_197M = 2 * HSR_RF_98M
HSR_RF_591M = 3 * HSR_RF_197M

HSR_BUNCH_LENGTH_23GeV  = 50.0 / const.c     # - cm rms
HSR_BUNCH_LENGTH_41GeV  = 11.0 / const.c     # - cm rms
HSR_BUNCH_LENGTH_100GeV = 7.0  / const.c     # - cm rms
HSR_BUNCH_LENGTH_275GeV = 6.0  / const.c     # - cm rms

# ---- ESR Frequencies ---- #
ESR_REV_FREQ            = 78194.34  # - Hz
ESR_RF_HARMONIC_NUMBER  = 7560
ESR_RF_591M             = ESR_RF_HARMONIC_NUMBER * ESR_REV_FREQ
ESR_BUNCH_LENGTH_5GeV   = 23.35     # - ps rms
ESR_BUNCH_LENGTH_10GeV  = 30.02     # - ps rms

# ---- EIS Frequencies ---- #

# -------------------------------- Signal Creation ------------------------------------- #

# --- Create a Gaussian Pulse
def gaussian_pulse(t: np.ndarray, sigma: float = 1.0, FWHM: float = None, rf: float = None, BW: float = 1) -> tuple[np.ndarray, float]:
    """Generate a Gaussian pulse. The pulse can be defined by its standard deviation (sigma), full width at half maximum (FWHM), or radio frequency (rf) and bandwidth (BW).
    
    Args:
        t (np.ndarray): Time array.
        sigma (float, optional): Standard deviation of the Gaussian pulse. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
    
    Returns:
        tuple[np.ndarray, float]: Normalized Gaussian pulse and its calculated standard deviation.
    """
    if rf is None:
        s = sigma if FWHM is None else FWHM / (2 * np.sqrt(2 * np.log(2)))
        gaus_pulse = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (t / s)**2)
        return gaus_pulse / np.max(gaus_pulse), s
    else:
        ref = np.power(10.0, -6 / 20.0)
        a = -(np.pi * rf * BW) ** 2 / (4.0 * np.log(ref))
        yenv = np.exp(-a * t**2)
        s_calc = calculate_pulse_width( yenv, t[1] - t[0])  # Optional: Calculate pulse width for verification
        return yenv, s_calc

# --- Create a Skew-Gaussian Pulse
def skew_gaus_pulse(t: np.ndarray, skew: float = 1.0, sigma: float = 1.0, FWHM: float = None, rf: float = None, BW: float = 1) -> tuple[np.ndarray, float]:
    """Generate a skewed Gaussian pulse.
    
    Args:
        t (np.ndarray): Time array.
        skew (float, optional): Skew factor. Defaults to 1.0.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
    
    Returns:
        tuple[np.ndarray, float]: Skewed Gaussian pulse and its standard deviation.
    """
    gaus, s = gaussian_pulse(t, sigma, FWHM, rf, BW)
    skew_factor = (1 - erf(-1 * (skew * t / (np.sqrt(2) * s))))
    skewed_pulse = gaus * skew_factor
    return skewed_pulse / np.max(skewed_pulse), s

# --- Create Gaussian Doublet
def gaus_doublet_pulse( t: np.ndarray, sigma: float = 1.0, FWHM: float = None, rf: float = None, BW: float = 1, skew: float = 1.0) -> tuple[np.ndarray, float]:
    """Generate a Gaussian doublet pulse.
    
    Args:
        t (np.ndarray): Time array.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
        skew (float, optional): Skew factor. Defaults to 1.0.
    
    Returns:
        tuple[np.ndarray, float]: Gaussian doublet pulse and its standard deviation.
    """
    gaus, s = gaussian_pulse(t, sigma, FWHM, rf, BW)
    return (-1*t/sigma**2)*gaus, s

# --- Create a Morlet Wavelet
def morlet_pulse(t: np.ndarray, f: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Generate a Morlet wavelet.
    
    Args:
        t (np.ndarray): Time array.
        f (float, optional): Frequency of the wavelet. Defaults to 1.0.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
    
    Returns:
        np.ndarray: Morlet wavelet.
    """
    return np.cos(2 * np.pi * f * t) * gaussian_pulse(t, sigma)[0]

# --- Create a Damped Sine Wave
def damped_sine_wave(t: np.ndarray, t0: float = 0.0, freq: float = 1.0, decay: float = 1.0, amp: float = 1.0, phi: float = 0.0) -> np.ndarray:
    """Generate a damped sine wave.
    
    Args:
        t (np.ndarray): Time array.
        t0 (float, optional): Start time of the sine wave. Defaults to 0.0.
        freq (float, optional): Frequency of the sine wave. Defaults to 1.0.
        decay (float, optional): Decay factor. Defaults to 1.0.
        amp (float, optional): Amplitude of the sine wave. Defaults to 1.0.
        phi (float, optional): Phase shift of the sine wave. Defaults to 0.0.
    
    Returns:
        np.ndarray: Damped sine wave.
    """
    dsw = np.exp(-decay * (t-t0)) * np.sin(2 * np.pi * freq * (t- t0) + phi)
    dsw[t < t0] = 0.0
    dswn = dsw / np.max(dsw)
    return amp*dswn

# --- Create a Cosine Square Pulse 
def cosine_square_pulse(t: np.ndarray, period: float, amplitude: float = 1.0, phi: float = 0) -> np.ndarray:
    """Generate a cosine square pulse.
    
    Args:
        t (np.ndarray): Time array.
        period (float): Period of the pulse.
        amplitude (float, optional): Amplitude of the pulse. Defaults to 1.0. 
        phi (float, optional): Signal phase in degrees. Defaults to 0.
    
    Returns:
        np.ndarray: Cosine square pulse.
    """
    cos_phi = phi*(np.pi/180)
    cos = np.cos((2*np.pi*t)/period+(np.pi+cos_phi))
    cos_sq = (cos-1)**2
    cos_sq /= cos_sq.max()   # Normalize the pulse to its maximum value
    tshift = -1*(cos_phi/(2*np.pi))*period
    cos_sq[ t > tshift+(0.5*period) ] = 0.0
    cos_sq[ t < tshift+(-0.5*period) ] = 0.0
    return cos_sq * amplitude

# --- Uniform Pulse
def uniform_pulse(t: np.ndarray, pulse_width: float, pulse_BW: int = 10, amplitude: float = 1.0) -> np.ndarray:
    """Generate a uniform pulse from summed sines. This provides a more realistic pulse shape for uniform beams compared to the square pulse.
    
    Args:
        t (np.ndarray): Time array.
        pulse_width (float): Width of the pulse in seconds.
        pulse_BW (int, optional): Pulse bandwidth. Use this to determine how flat and sharp the sides. Defaults to 10.
        amplitude (float, optional): Amplitude of the pulse. Defaults to 1.0.
    
    Returns:
        np.ndarray: Uniform pulse.
    """
    # --- Create the pulse 
    test_sig = np.sin( 2*np.pi*t/(2*pulse_width) )
    for i in np.arange(2,pulse_BW,1):
        if i%2:
            test_sig += (1/i)*np.sin( (2*np.pi*t)*(i/(2*pulse_width)) ) 
    
    # --- Apply Gaussian Smoothing
    test_sig = gaus_smooth(test_sig, int(t.size*0.05), int(t.size*0.07))
    
    # --- Normalize the pulse to its maximum value
    test_sig /= np.max(test_sig)
    
    return test_sig * amplitude

# --- Create a Square Pulse
def square_pulse(t: np.ndarray, period: float, amplitude: float = 1.0) -> np.ndarray:
    """Generate a square pulse.
    
    Args:
        t (np.ndarray): Time array.
        period (float): Period of the pulse.
        amplitude (float, optional): Amplitude of the pulse. Defaults to 1.0.
    
    Returns:
        np.ndarray: Square pulse.
    """
    pulse = np.zeros(t.size)
    pulse[ np.abs(t) < period/2 ] = 1.0
    return pulse * amplitude

# --- Create a Triangle Pulse
def triangle_pulse(t: np.ndarray, period: float, amplitude: float = 1.0) -> np.ndarray:
    """Generate a triangle pulse.
    
    Args:
        t (np.ndarray): Time array.
        period (float): Period of the pulse.
        amplitude (float, optional): Amplitude of the pulse. Defaults to 1.0.
    
    Returns:
        np.ndarray: Triangle pulse.
    """
    tri = -1*sg.sawtooth(2 * np.pi * t / period, width=0.5)+1
    tri[ np.abs(t) > period/2 ] = 0.0
    tri /= tri.max()
    return tri * amplitude

# --------------------------- Waveform Generation --------------------------- #

# --- Create a Sine Wave
def sine_wave(t: np.ndarray, frequency: float = 1.0, phase: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
    """Generate a sine wave.
    
    Args:
        t (np.ndarray): Time array.
        frequency (float, optional): Frequency of the sine wave. Defaults to 1.0.
        phase (float, optional): Phase shift of the sine wave. Defaults to 0.0.
        amplitude (float, optional): Amplitude of the sine wave. Defaults to 1.0.
    
    Returns:
        np.ndarray: Sine wave.
    """
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

# --- Create a Square Wave
def square_wave(t: np.ndarray, frequency: float = 1.0, duty_cycle: float = 0.5, amplitude: float = 1.0) -> np.ndarray:
    """Generate a square wave.
    
    Args:
        t (np.ndarray): Time array.
        frequency (float, optional): Frequency of the square wave. Defaults to 1.0.
        duty_cycle (float, optional): Duty cycle of the square wave. Defaults to 0.5.
        amplitude (float, optional): Amplitude of the square wave. Defaults to 1.0.
    
    Returns:
        np.ndarray: Square wave.
    """
    return amplitude * sg.square(2 * np.pi * frequency * t, duty=duty_cycle)    

# --- Create a Triangle Wave
def triangle_wave(t: np.ndarray, frequency: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """Generate a triangle wave.
    
    Args:
        t (np.ndarray): Time array.
        frequency (float, optional): Frequency of the triangle wave. Defaults to 1.0.
        amplitude (float, optional): Amplitude of the triangle wave. Defaults to 1.0.
    
    Returns:
        np.ndarray: Triangle wave.
    """
    return amplitude * sg.sawtooth(2 * np.pi * frequency * t, width=0.5)

# --- Create a dc waveform with linear transitions
def create_transition_array(length: int, values: tuple, change_points: tuple, transition_lengths: tuple) -> np.ndarray:
    """
    Create an array with transitions to different values based on the provided parameters.

    Args:
        length (int): Number of turns in the array.
        values (tuple): Tuple containing the values to transition to.
        change_points (tuple): Tuple containing the turn numbers where the array changes values.
        transition_lengths (tuple): Tuple containing the number of turns it takes to transition to the new value.

    Returns:
        np.ndarray: Array with transitions to different values.
    """
    if len(values) != len(change_points) + 1 or len(change_points) != len(transition_lengths):
        raise ValueError("Invalid input lengths. Ensure len(values) = len(change_points) + 1 and len(change_points) = len(transition_lengths).")

    array = np.zeros(length)
    current_value = values[0]
    array[:change_points[0]] = current_value

    for i in range(len(change_points)):
        start = change_points[i]
        end = start + transition_lengths[i]
        next_value = values[i + 1]
        transition = np.linspace(current_value, next_value, transition_lengths[i])
        array[start:end] = transition
        current_value = next_value
        if i < len(change_points) - 1:
            array[end:change_points[i + 1]] = current_value
        else:
            array[end:] = current_value

    return array

# -------------------------------- Signal Modulations ------------------------------------- #

# --- Modulating Gaussian Pulses
def mod_gaus_pulse(t: np.ndarray, gF: float, gBW: float, modF: list[float], gFAmp: float = 1.0, fAmp: float = 1.0, gRef: float = -6) -> np.ndarray:
    """Modulate a Gaussian pulse with multiple frequencies.
    
    Args:
        t (np.ndarray): Time array.
        gF (float): Gaussian pulse frequency.
        gBW (float): Gaussian bandwidth.
        modF (list[float]): List of modulation frequencies.
        gFAmp (float, optional): Amplitude of the Gaussian. Defaults to 1.0.
        fAmp (float, optional): Amplitude of the modulation frequencies. Defaults to 1.0.
        gRef (float, optional): Reference power level. Defaults to -6.
    
    Returns:
        np.ndarray: Modulated Gaussian waveform.
    """
    ref = np.power(10.0, gRef / 20.0)
    a = -(np.pi * gF * gBW) ** 2 / (4.0 * np.log(ref))
    yenv = np.exp(-a * t * t)
    mod = gFAmp * np.sin(2 * np.pi * gF * t)
    for f, amp in zip(modF, fAmp if isinstance(fAmp, list) else [fAmp] * len(modF)):
        mod *= amp * np.sin(2 * np.pi * f * t)
    mod += np.sqrt(gFAmp**2 + np.dot(fAmp, fAmp) if isinstance(fAmp, list) else fAmp**2)
    return yenv * mod / np.max(np.abs(yenv * mod))

# --- Modulating Cosine Square Pulses
def mod_cos_sq_pulse(t: np.ndarray, period: float, modF: float, pulsAmp: float = 1.0, modAmp: float = 1.0, phi: float = 0.0) -> np.ndarray:
    """Modulate a cosine square pulse with multiple frequencies.
    
    Args:
        t (np.ndarray): Time array.
        period (float): Period of the pulse.
        modF (float): Modulation frequency.
        modAmp (float, optional): Amplitude of the modulation frequencies. Defaults to 1.0.
        phi (float, optional): Signal phase in degrees. Defaults to 0.
    
    Returns:
        np.ndarray: Modulated cosine square waveform.
    """
    csp = cosine_square_pulse(t, period, amplitude=pulsAmp, phi=phi)
    csp_sin = np.sin((1/period)*2*np.pi*t)
    
    mod_sin = modAmp*np.sin((modF)*2*np.pi*t)

    mod_csp = (csp_sin*mod_sin+np.sqrt(modAmp**2+pulsAmp**2))
    mod_csp *= csp
    mod_csp /= mod_csp.max()

    return mod_csp

# --- Modulated pulse train
def mod_pulse_train(t: np.ndarray, rf: float, pulse: np.ndarray, pulse_time: np.ndarray, bunches: int, modF: float, modAmp: float) -> np.ndarray:
    """Modulate a pulse train with a given frequency and amplitude.
    
    Args:
        t (np.ndarray): Time array.
        rf (float): Radio frequency of pulses.
        pulse (np.ndarray): Single pulse waveform.
        pulse_time (np.ndarray): Time array for a single pulse.
        bunches (int): Number of pulses in the train.
        modF (float): Modulation frequency.
        modAmp (float): Modulation amplitude.
    
    Returns:
        np.ndarray: Modulated pulse train waveform.
    """
    _, pulse_train = create_pulse_train(rf, pulse, pulse_time, bunches)
    mod = (np.sin(2 * np.pi * modF * t)+1)
    mod = mod/mod.max() * modAmp            # Normalize the modulation to the amplitude
    return pulse_train * mod

# -------------------------------- Signal Pulse Trains ------------------------------------- #

# --- Create a Pulse Train
def create_pulse_train(rf: float, pulse: np.ndarray, pulse_time: np.ndarray, bunches: int, pad_train=None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a train of pulses.
    
    Args:
        rf (float): Radio frequency of pulses.
        pulse (np.ndarray): Single pulse waveform.
        pulse_time (np.ndarray): Time array for a single pulse.
        bunches (int): Number of pulses in the train.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Time array and pulse train waveform.
    """
    #TODO create_pulse_train - a full period before and after the train is zero, make optional.
    # --- New arrays so originals are not changed
    psig = np.copy(pulse)
    ptime = np.copy(pulse_time)
    
    Fs = 1 / (ptime[1] - ptime[0])
    pulse_period = 1 / rf
    
    # Check if pulse_time is smaller than pulse_period
    if (ptime[-1] - ptime[0]) < pulse_period:
        # Calculate the number of samples to zero-pad
        num_zeros = int((pulse_period - (ptime[-1] - ptime[0])) * Fs)
        
        # Zero-pad the pulse
        psig = np.pad(psig, num_zeros//2, mode='constant')
        
        # Create new pulse_time
        ptime = np.linspace( -0.5*pulse_period, 0.5*pulse_period, psig.size )

    temp = psig[ptime > -pulse_period / 2]
    temp_time_ary = ptime[ptime > -pulse_period / 2]
    temp = temp[temp_time_ary < pulse_period / 2]
    
    # --- Create the pulse train
    # --- --- pad if option there... 
    if pad_train is None:
        pulse_ary = temp
    elif isinstance(pad_train, int):
        pulse_ary = np.concatenate((np.zeros(temp.size*pad_train), temp))
    elif isinstance(pad_train, tuple) and len(pad_train) == 2:
        pulse_ary = np.concatenate((np.zeros(temp.size * pad_train[0]), temp))
    else:
        raise ValueError("pad_train must be None, an int, or a tuple of two ints")
    
    # --- --- build train
    for _ in range(bunches - 1):
        pulse_ary = np.concatenate((pulse_ary, temp))
    
    # --- --- pad if option there...
    if isinstance(pad_train, tuple) and len(pad_train) == 2:
        pulse_ary = np.concatenate((pulse_ary, np.zeros(temp.size*pad_train[1])))
    
    return np.linspace(0.0, pulse_ary.size / Fs, pulse_ary.size), pulse_ary

# --- Create a Pulse Train with Jitter
#TODO pulse train jitter
def create_pulse_train_jitter(rf: float, pulse: np.ndarray, pulse_time: np.ndarray, bunches: int, jitter: float) -> tuple[np.ndarray, np.ndarray]:
    """UNDER DEVELOPMENT. Generate a train of pulses with jitter.  
    
    Args:
        rf (float): Radio frequency of pulses.
        pulse (np.ndarray): Single pulse waveform.
        pulse_time (np.ndarray): Time array for a single pulse.
        bunches (int): Number of pulses in the train.
        jitter (float): Jitter in seconds.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Time array and pulse train waveform.
    """
    Fs = 1 / (pulse_time[1] - pulse_time[0])
    pulse_period = 1 / rf
    temp = pulse[pulse_time > -pulse_period / 2]
    temp = temp[pulse_time < pulse_period / 2]
    pulse_ary = np.concatenate((np.zeros(temp.size), temp))
    for _ in range(bunches - 1):
        pulse_ary = np.concatenate((pulse_ary, temp))
    pulse_ary = np.concatenate((pulse_ary, np.zeros(temp.size)))
    ''' NEED TO ADJUST HERE - try moving up in the function '''
    jitter = int(jitter * Fs)
    
    pulse_ary = np.roll(pulse_ary, jitter)
    return np.linspace(0.0, pulse_ary.size / Fs, pulse_ary.size), pulse_ary

# -------------------------------- Other ------------------------------------- #

# --- Calculate RMS Pulse Width
def calculate_pulse_width(pulse_sig: np.ndarray, tp: float) -> float:
    """Calculate the pulse width of a given pulse signal.
    
    Args:
        pulse_sig (np.ndarray): Signal array of a single pulse.
        tp (float): Time period.
    
    Returns:
        float: Calculated RMS pulse width in nanoseconds.
    """
    peaks, _ = sg.find_peaks(pulse_sig)
    half = sg.peak_widths(pulse_sig, peaks)
    pulse_width = (half[0] / (2 * np.sqrt(2 * np.log(2))))[0]
    return pulse_width * tp * 1e9

# --- Calculate signal spectrum
def get_spectrum(sig: np.ndarray, Fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Get the spectrum of a signal.
    
    Args:
        sig (np.ndarray): Signal array.
        Fs (float): Sampling frequency.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Frequency array and spectrum.
    """
    #x = len(sig)
    yf = fftshift(fft(sig))
    f = fftshift( fftfreq(sig.size, (1 / Fs)) )

    return f, yf

# --- Export to SPICE PWL
def export_to_spice_pwl(time_array: np.ndarray, value_array: np.ndarray, filename: str) -> None:
    """
    Exports time and value arrays into a text file formatted for SPICE PWL.

    Args:
        time_array (np.ndarray): Array representing time values.
        value_array (np.ndarray): Array representing source values.
        filename (str): Name of the output file.
    """
    if len(time_array) != len(value_array):
        raise ValueError("Time array and value array must have the same length.")

    with open(filename, 'w') as file:
        for t, v in zip(time_array, value_array):
            file.write(f"{t}\t{v}\n")

# --- Gaussian Smoothing
def gaus_smooth(sig: np.ndarray, FWHM: float, win_size: int, debug: bool = False) -> np.ndarray:
    """Apply Gaussian smoothing to a signal.
    
    Args:
        sig (np.ndarray): Input signal array.
        FWHM (float): Full width at half maximum of the Gaussian.
        win_size (int): Size of the Gaussian window.
        debug (bool, optional): Flag to plot the Gaussian profile for debugging. Defaults to False.
    
    Returns:
        np.ndarray: Smoothed signal array.
    """
    if win_size <= 0:
        raise ValueError("win_size must be a positive integer.")
    if FWHM <= 0:
        raise ValueError("FWHM must be a positive float.")

    # --- Copy input signal and set edge effects to unsmoothed signal
    temp_sig = np.array( sig )
    smooth_sig = np.copy( temp_sig )
    
    # --- define Gaussian and normalize
    x = np.linspace(-win_size//2,win_size//2, win_size)
    gf = np.exp( -1*(4*np.log(2)*x**2) / (FWHM**2) )
    gf /= gf.sum()
    
    if debug:
        print("Gauss Sum = {}".format(gf.sum()))
        plt.figure()
        plt.plot(x, gf)
        plt.title("DEBUG: Gaussian Profile")
        plt.xlabel("Samples")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.show()

    # --- Apply across the signl
    for i in range(win_size//2,temp_sig.size-win_size//2):
        temp = temp_sig[(i-win_size//2):(i+win_size//2)] * gf
        smooth_sig[i] = temp.sum()
    
    return smooth_sig

import numpy as np

def gamma_to_beta(gamma):
    """
    Convert Lorentz factor gamma to velocity ratio beta = v/c.

    Parameters:
        gamma (float or array-like): Lorentz factor (must be >= 1)

    Returns:
        beta (float or array-like): Velocity as a fraction of the speed of light
    """
    gamma = np.asarray(gamma)
    if np.any(gamma < 1):
        raise ValueError("Gamma must be >= 1")
    beta = np.sqrt(1 - 1 / gamma**2)
    return beta