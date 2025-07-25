import numpy as np
import scipy.signal as sg
import scipy.constants as const
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.special import erf
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple, List


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

# ---- HSR Bunch Lengths ---- #
# ---- ---- PP ---- ---- #
HSR_PP_BUNCH_LENGTH_23GeV  = 50.0e-2 / const.c     # - s rms (formula is given in cm)
HSR_PP_BUNCH_LENGTH_41GeV  = 7.5e-2 / const.c     # - s rms (formula is given in cm)
HSR_PP_BUNCH_LENGTH_100GeV = 7.0e-2  / const.c     # - s rms (formula is given in cm)
HSR_PP_BUNCH_LENGTH_275GeV = 6.0e-2  / const.c     # - s rms (formula is given in cm)
# ---- ---- AU ---- ---- #
HSR_AU_BUNCH_LENGTH_10GeV  = 125.0e-2 / const.c     # - s rms (formula is given in cm)
HSR_AU_BUNCH_LENGTH_41GeV  = 11.6e-2  / const.c     # - s rms (formula is given in cm)
HSR_AU_BUNCH_LENGTH_110GeV = 7.0e-2   / const.c     # - s rms (formula is given in cm)

# ---- ESR Frequencies ---- #
ESR_REV_FREQ            = 78194.34  # - Hz
ESR_RF_HARMONIC_NUMBER  = 7560
ESR_RF_591M             = ESR_RF_HARMONIC_NUMBER * ESR_REV_FREQ
ESR_BUNCH_LENGTH_5GeV   = 23.35e-12     # - s rms
ESR_BUNCH_LENGTH_18GeV  = 30.02e-12     # - s rms

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
def cosine_square_pulse(t: np.ndarray, pulse_width: float, pw_type: str = 'fixed', amplitude: float = 1.0, phi: float = 0) -> np.ndarray:
    """Generate a cosine square pulse.
    
    Args:
        t (np.ndarray): Time array.
        pulse_width (float): Pulse width in seconds. Width is configurable by optional argument `pw_type`.
        amplitude (float, optional): Amplitude of the pulse. Defaults to 1.0. 
        phi (float, optional): Signal phase in degrees. Defaults to 0.
        pw_type (str, optional): Pulse width type. Defaults to 'fixed'.
    
    Returns:
        np.ndarray: Cosine square pulse.
    """
    # --- Determine the pulse width based on input arguments
    pwType = pw_type.lower()
    if pwType == 'fwhm':
        period = ( pulse_width / (2 * np.sqrt(2 * np.log(2)) ) ) * 7.0 # Convert to RMS then to 99.99% width
    elif pwType == 'rms':
        period = pulse_width * 7.0       # Convert to 99.99% width
    else:
        period = pulse_width
    
    # --- Create the cosine square pulse
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
def create_transition_array(
        length: int,
        values: Tuple[float,float],
        change_points: Union[Tuple[int,int], List[int], int],
        transition_lengths: Union[Tuple[int,int], List[int], int],
        ) -> np.ndarray:
    """
    Create an array with transitions to different values based on the provided parameters.

    Args:
        length (int): Number of turns in the array.
        values (tuple): Tuple containing the values to transition to.
        change_points (tuple, list, or int): Turn numbers where the array changes values.
        transition_lengths (tuple, list, or int): Number of turns it takes to transition to the new value.

    Returns:
        np.ndarray: Array with transitions to different values.
    """

    # Normalize change_points and transition_lengths to lists
    if isinstance(change_points, (int, float)):
        change_points = [change_points] * (len(values) - 1)
    elif isinstance(change_points, (tuple, list)):
        change_points = list(change_points)

    if isinstance(transition_lengths, (int, float)):
        transition_lengths = [transition_lengths] * (len(values) - 1)
    elif isinstance(transition_lengths, (tuple, list)):
        transition_lengths = list(transition_lengths)

    # Validate input lengths
    if len(values) != len(change_points) + 1 or len(change_points) != len(transition_lengths):
        raise ValueError("Invalid input lengths. Ensure len(values) = len(change_points) + 1 and len(change_points) = len(transition_lengths).")

    # Initialize the array
    array = np.zeros(length)
    current_value = values[0]

    # Fill up to first change point
    array[:change_points[0]] = current_value

    # Apply transitions
    for i in range(len(change_points)):
        start = change_points[i]
        end = min(start + transition_lengths[i], length)
        next_value = values[i + 1]

        # Create linear transition
        transition = np.linspace(current_value, next_value, end - start, endpoint=False)
        array[start:end] = transition

        current_value = next_value

        # Fill the next flat section (if any)
        if i < len(change_points) - 1:
            next_cp = change_points[i + 1]
            if end < next_cp:
                array[end:next_cp] = current_value
        else:
            # Fill the remainder of the array
            if end < length:
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
def mod_cos_sq_pulse(t: np.ndarray, pulse_width: float, baseF: float, modF: float, pw_type: str = 'fixed', pulsAmp: float = 1.0, modAmp: float = 1.0, phi: float = 0.0) -> np.ndarray:
    """Modulate a cosine square pulse with multiple frequencies.
    
    Args:
        t (np.ndarray): Time array.
        pulse_width (float): Pulse width in seconds. Width is configurable by optional argument `pw_type`.
        baseF (float): Base frequency of the cosine square pulse.
        modF (float): Modulation frequency.
        pw_type (str, optional): Pulse width type. Defaults to 'fixed'.
        pulsAmp (float, optional): Amplitude of the cosine square pulse. Defaults to 1.0.
        modAmp (float, optional): Amplitude of the modulation frequencies. Defaults to 1.0.
        phi (float, optional): Signal phase in degrees. Defaults to 0.
    
    Returns:
        np.ndarray: Modulated cosine square waveform.
    """
    csp = cosine_square_pulse(t, pulse_width, pw_type=pw_type, amplitude=pulsAmp, phi=phi)
    csp_sin = np.sin((baseF)*2*np.pi*t)
    
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

# ------------------- Plotting functions ------------------- #
def plot_gaus_pulses(
    t: np.ndarray,
    pulses: Union[Dict[str, np.ndarray], np.ndarray],
    freq: np.ndarray,
    pulse_spec: Union[Dict[str, np.ndarray], np.ndarray],
    title: str = None,
    ax1_xlabel: str = None,
    ax1_ylabel: str = None,
    ax1_xlim: Tuple[float, float] = None,
    ax1_ylim: Tuple[float, float] = None,
    ax2_xlabel: str = None,
    ax2_ylabel: str = None,
    ax2_xlim: Tuple[float, float] = None,
    ax2_ylim: Tuple[float, float] = None,
    grid: bool = True,
    legend: bool = True,
    loglog: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100,
    **kwargs
) -> None:
    """
    Plot Gaussian pulses in time and frequency domains.

    Parameters:
    t (np.ndarray): Time array
    pulses (Union[Dict[str, np.ndarray], np.ndarray]): Dictionary of pulse values or single pulse array
    freq (np.ndarray): Frequency array
    pulse_spec (Union[Dict[str, np.ndarray], np.ndarray]): Dictionary of pulse spectra or single pulse spectrum array
    title (str, optional): Plot title. Defaults to None.
    ax1_xlabel (str, optional): X-axis label for time plot. Defaults to None.
    ax1_ylabel (str, optional): Y-axis label for time plot. Defaults to None.
    ax1_xlim (Tuple[float, float], optional): X-axis limits for time plot. Defaults to None.
    ax1_ylim (Tuple[float, float], optional): Y-axis limits for time plot. Defaults to None.
    ax2_xlabel (str, optional): X-axis label for frequency plot. Defaults to None.
    ax2_ylabel (str, optional): Y-axis label for frequency plot. Defaults to None.
    ax2_xlim (Tuple[float, float], optional): X-axis limits for frequency plot. Defaults to None.
    ax2_ylim (Tuple[float, float], optional): Y-axis limits for frequency plot. Defaults to None.
    grid (bool, optional): Show grid. Defaults to True.
    legend (bool, optional): Show legend. Defaults to True.
    loglog (bool, optional): Use log-log scale for frequency plot. Defaults to False.
    figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 6).
    dpi (int, optional): Figure DPI. Defaults to 100.
    **kwargs: Additional keyword arguments for plot customization

    Returns:
    None
    """
    tt = t*1e9  # Convert time to nanoseconds
    ff = freq*1e-9  # Convert frequency to GHz

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)

    if isinstance(pulses, dict):
        for p in pulses.keys():
            # sigma = calculate_pulse_width(pulses[p], t[1] - t[0]) / 1e9
            # ax1.plot(tt, pulses[p], label=label or f"$\\sigma$={sigma*const.c*100:.2f}cm")
            ax1.plot(tt, pulses[p], label=p)
            if loglog:
                ax2.loglog(ff, np.abs(pulse_spec[p]), label=p)
                        #    label=label or f"$\\sigma$={sigma*const.c*100:.2f}cm")
                            
            else:
                ax2.plot(ff, np.abs(pulse_spec[p]),label=p)
                        #  label=label or f"$\\sigma$={sigma*const.c*100:.2f}cm")
    else:
        # sigma = calculate_pulse_width(pulses, t[1] - t[0]) / 1e9
        ax1.plot(tt, pulses, )
                #  label=label or f"$\\sigma$={sigma*const.c*100:.2f}cm")
        if loglog:
            ax2.loglog(ff, np.abs(pulse_spec), )
                    #    label=label or f"$\\sigma$={sigma*const.c*100:.2f}cm")
        else:
            ax2.plot(ff, np.abs(pulse_spec), )
                    #  label=label or f"$\\sigma$={sigma*const.c*100:.2f}cm")
            
    ax1.grid(axis="both", which="both" if grid else "none", ls='--', lw=0.5, color='gray')
    ax1.set_xlabel(ax1_xlabel or "Time [ns]")
    ax1.set_ylabel(ax1_ylabel or "Normalized Amplitude")
    ax1.set_title(title+", Time" or 'Ideal Beam Pulse, Time')
    if ax1_xlim:
        ax1.set_xlim(ax1_xlim)
    if ax1_ylim:
        ax1.set_ylim(ax1_ylim)
    if legend:
        ax1.legend()

    ax2.grid(axis="both", which="both" if grid else "none", ls='--', lw=0.5, color='gray')
    ax2.set_xlabel(ax2_xlabel or "Freq [GHz]")
    ax2.set_ylabel(ax2_ylabel or "Magnitude")
    ax2.set_title(title+", Freq" or 'Ideal Beam Pulse, Freq.')
    if ax2_xlim:
        ax2.set_xlim(ax2_xlim)
    if ax2_ylim:
        ax2.set_ylim(ax2_ylim)
    if legend:
        ax2.legend()

    plt.tight_layout()
    plt.show()

# --- More Fancy Plotting
#TODO : Clean up - This compares the ideal pulse with the filtered pulse
def plot_gaus_compare( org, flt, lbl):
    colors = ["k","b","g","m","c","r"]
    # --- check if flt is a tuple 
    if (type(lbl) is tuple):
        fig, ax = plt.subplots(nrows=len(lbl), ncols=1)
        for a,f,l in zip( ax, flt, lbl ):
            for p,c in zip(org,colors):
                if p!="Time":
                    a.plot(org["Time"]*1e9, org[p], color=c, label="$\\sigma$="+p)
                    a.plot(f["Time"]*1e9, f[p], color=c, linestyle="dashdot", label="$\\sigma$="+p+" "+l)
                a.grid(axis="both",which="both")
                a.set_xlabel("Time [ns]")
                a.set_ylabel("Normalized Amplitude")
                a.set_title('Ideal Beam Pulse vs '+l)
                a.legend()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for p, c in zip(org, colors):
            if p!="Time":
                ax.plot(org["Time"]*1e9, org[p], color=c, label='$\\sigma$='+p)
                ax.plot(flt["Time"]*1e9, flt[p], color=c, linestyle="dashdot", label="$\\sigma$="+p+" "+lbl)
            ax.grid(axis="both",which="both")
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Normalized Amplitude")
            ax.set_title('Ideal Beam Pulse vs '+lbl)
            ax.legend()
    plt.tight_layout()
    plt.show() 

# --- Waterfall plot generator
def waterfall_plot(waveforms: list[np.ndarray], num_waveforms_to_plot: int = 50, time_array: np.ndarray = None) -> None:
    """
    Creates a waterfall plot of a subset of waveforms from a list.

    Args:
        waveforms: A list of numpy arrays, where each array represents a waveform.
        num_waveforms_to_plot: The number of evenly spaced waveforms to display.
        time_array: An optional array of time values for the waveforms.

    Raises:
        ValueError: If num_waveforms_to_plot exceeds the total number of waveforms.

    Returns:
        None
    """

    num_total_waveforms = len(waveforms)

    if num_waveforms_to_plot > num_total_waveforms:
        raise ValueError("Number of waveforms to plot cannot exceed the total number of waveforms.")

    indices = np.linspace(0, num_total_waveforms - 1, num_waveforms_to_plot, dtype=int)
    selected_waveforms = [waveforms[i] for i in indices]

    plt.figure(figsize=(10, 6))

    for i, waveform in enumerate(selected_waveforms):
        if time_array is not None:
            time = np.arange(len(waveform))  # Assuming time is just sample index.
        else:
            time = np.array(time_array)
        plt.plot(time, waveform + i * 5, linewidth=0.75, color='b') # Add offset to each waveform

    plt.title("Waterfall Plot of Waveforms")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Amplitude + Offset")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()