import numpy as np
import scipy.signal as sg
import scipy.constants as const
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
def gaussian_pulse(
    t: np.ndarray,
    sigma: float = 1.0,
    FWHM: float = None,
    rf: float = None,
    BW: float = 1,
    phase_shift: float = 0.0,
    phase_freq: float = None
) -> np.ndarray:
    """
    Generate a Gaussian pulse with optional phase shift based on frequency.

    Args:
        t (np.ndarray): Time array.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
        phase_shift (float, optional): Phase shift in degrees. Defaults to 0.0.
        phase_freq (float, optional): Frequency in Hz for phase shift calculation. Defaults to None.

    Returns:
        np.ndarray: Gaussian pulse with optional phase shift.
    """
    t_shift = (phase_shift / 360.0) / phase_freq if phase_freq else 0.0
    t = t - t_shift

    if rf is None:
        # Calculate the standard deviation from the FWHM if given
        s = sigma if FWHM is None else FWHM / (2 * np.sqrt(2 * np.log(2)))
        gaus_pulse = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (t / s)**2)
        return gaus_pulse / np.max(gaus_pulse), s
    else:
        # Create gaussian envelope from rf and the bandwidth if given
        ref = np.power(10.0, -6 / 20.0)
        a = -(np.pi * rf * BW) ** 2 / (4.0 * np.log(ref))
        yenv = np.exp(-a * t**2)
        return yenv / np.max(yenv), calculate_pulse_width(yenv,t[1]-t[0])

# --- Create a Skew-Gaussian Pulse
def skew_gaus_pulse(
    t: np.ndarray,
    skew: float = 1.0,
    sigma: float = 1.0,
    FWHM: float = None,
    rf: float = None,
    BW: float = 1,
    phase_shift: float = 0.0,
    phase_freq: float = None
) -> np.ndarray:
    """
    Generate a skewed Gaussian pulse with optional phase shift based on frequency.

    Args:
        t (np.ndarray): Time array.
        skew (float, optional): Skew factor. Defaults to 1.0.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
        phase_shift (float, optional): Phase shift in degrees. Defaults to 0.0.
        phase_freq (float, optional): Frequency in Hz for phase shift calculation. Defaults to None.

    Returns:
        np.ndarray: Skewed Gaussian pulse with optional phase shift.
    """
    gaus, s = gaussian_pulse(t, sigma, FWHM, rf, BW, phase_shift, phase_freq)
    # include time shift
    t_shift = (phase_shift / 360.0) / phase_freq if phase_freq else 0.0
    t = t - t_shift
    # Calculate the skew factor using the error function
    skew_factor = 1 - erf(-1 * (skew * t / (np.sqrt(2) * s)))
    # Apply the skew factor to the Gaussian pulse
    skewed_pulse = gaus * skew_factor
    # Normalize the skewed pulse to have a maximum amplitude of 1
    return skewed_pulse / np.max(skewed_pulse)

# --- Create Gaussian Doublet
def gaus_doublet_pulse( 
        t: np.ndarray, 
        sigma: float = 1.0, 
        FWHM: float = None, 
        rf: float = None, 
        BW: float = 1, 
        phase_shift: float = 0.0, 
        phase_freq: float = None,
) -> np.ndarray:
    """Generate a Gaussian doublet pulse.
    
    Args:
        t (np.ndarray): Time array.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
        phase_shift (float, optional): Phase shift in degrees. Defaults to 0.0.
        phase_freq (float, optional): Frequency in Hz for phase shift calculation. Defaults to None.
    
    Returns:
       np.ndarray: Gaussian doublet pulse with optional phase shift.
    """
    gaus, s = gaussian_pulse(t, sigma, FWHM, rf, BW, phase_shift, phase_freq)
    t_shift = (phase_shift / 360.0) / phase_freq if phase_freq else 0.0
    t = t - t_shift
    sig = (-1*t/s**2)*gaus
    return sig / np.max( sig )

# --- Create a Morlet Wavelet
def morlet_pulse(
        t: np.ndarray, 
        f: float = 1.0, 
        sigma: float = 1.0,
        FWHM: float = None, 
        rf: float = None, 
        BW: float = 1, 
        phase_shift: float = 0.0, 
        phase_freq: float = None,
) -> np.ndarray:
    """Generate a Morlet wavelet.
    
    Args:
        t (np.ndarray): Time array.
        f (float, optional): Frequency of the wavelet. Defaults to 1.0.
        sigma (float, optional): Standard deviation. Defaults to 1.0.
        FWHM (float, optional): Full width at half maximum. Defaults to None.
        rf (float, optional): Radio frequency. Defaults to None.
        BW (float, optional): Bandwidth. Defaults to 1.
        phase_shift (float, optional): Phase shift in degrees. Defaults to 0.0.
        phase_freq (float, optional): Frequency in Hz for phase shift calculation. Defaults to None.
    
    Returns:
        np.ndarray: Morlet wavelet.
    """
    gaus, s = gaussian_pulse(t, sigma, FWHM, rf, BW, phase_shift, phase_freq)
    sig = np.cos(2 * np.pi * f * t) * gaus
    return sig / np.max( abs( sig ) )

# --- Create a Damped Sine Wave
def damped_sine_wave(
        t: np.ndarray, 
        t0: float = 0.0, 
        freq: float = 1.0, 
        decay: float = 1.0, 
        amp: float = 1.0, 
        phi: float = 0.0
) -> np.ndarray:
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
def cosine_square_pulse(
    t: np.ndarray,
    pulse_width: float,
    pw_type: str = 'fixed',
    amplitude: float = 1.0,
    phase_freq: float = None,
    phase_shift: float = 0.0,
    degrees: bool = False
) -> np.ndarray:
    """
    Generate a cosine-squared pulse with optional frequency-based phase shift.

    Args:
        t (np.ndarray): Time array.
        pulse_width (float): Pulse width in seconds.
        pw_type (str, optional): Pulse width type. 'fixed', 'fwhm', or 'rms'. Defaults to 'fixed'.
        amplitude (float, optional): Amplitude of the pulse. Defaults to 1.0.
        phase_freq (float, optional): Frequency in Hz for phase shift calculation. If None, uses internal cosine width.
        phase_shift (float, optional): Phase shift (in degrees if degrees=True, else in radians).
        degrees (bool, optional): Whether phase_shift is in degrees. Defaults to False.

    Returns:
        np.ndarray: Cosine-squared pulse.
    """
    # Convert width to full extent (99.99%)
    pw_type = pw_type.lower()
    if pw_type == 'fwhm':
        period = (pulse_width / (2 * np.sqrt(2 * np.log(2)))) * 7.0
    elif pw_type == 'rms':
        period = pulse_width * 7.0
    else:
        period = pulse_width

    # Handle phase shift
    phase = np.deg2rad(phase_shift) if degrees else phase_shift

    # Frequency-based phase shift (adjust waveform in time)
    if phase_freq is not None:
        # time shift = phase / (2πf)
        t_shift = -phase / (2 * np.pi * phase_freq)
    else:
        # fallback: use phase shift relative to pulse width period
        t_shift = -phase / (2 * np.pi) * period

    # Cosine-squared profile
    cos_arg = (2 * np.pi * (t - t_shift)) / period + np.pi
    waveform = (np.cos(cos_arg) - 1) ** 2
    waveform /= np.max(waveform)

    # Apply windowing to keep only inside ±0.5*period
    mask = np.abs(t - t_shift) <= (0.5 * period)
    waveform[~mask] = 0.0

    return amplitude * waveform

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
    csp = cosine_square_pulse(t, pulse_width, pw_type=pw_type, amplitude=pulsAmp, phase_shift=phi, degrees=True)
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

# --- Create a Pulse Train of Arbitrary Pulses
def create_arb_pulse_train(
        rf: float, 
        pulse: np.ndarray, 
        pulse_time: np.ndarray, 
        bunches: int, 
        pad_train=None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a train of arbitrary pulses. 
    
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

# --- Create a Pulse Train of standard Pulses
def create_pulse_train(
    rf: float,
    bunches: int,
    pulse_width: float,
    phase_noise: float = 0.0,
    pulse_type: str = "cos",
    dt: float = 100e-12,
    pad_begin: int = None,
    pad_end: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:

    # --- Create Time Array
    """
    Generate a pulse train with optional phase noise and padding.

    Args:
        rf (float): Radio frequency of the pulses.
        bunches (int): Number of pulses in the train.
        pulse_width (float): Width of each pulse in seconds.
        phase_noise (float, optional): Maximum phase noise deviation in degrees. Defaults to 0.0.
        pulse_type (str, optional): Shape of the pulse. Defaults to 'cos'.
        dt (float, optional): Time step resolution in seconds. Defaults to 100e-12.
        pad_begin (int, optional): Number of zero-padded pulses at the beginning. Defaults to None.
        pad_end (int, optional): Number of zero-padded pulses at the end. Defaults to None.
        **kwargs: Additional keyword arguments for pulse waveform generation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Time array and pulse train waveform.
    """

    time_resolution = int( (1/rf) // dt )
    time_resolution = max( time_resolution, 1000 )

    test_time = np.linspace( -0.5/rf, 0.5/rf, time_resolution )
    dtt = test_time[1] - test_time[0]

    # --- Check for padding
    if pad_begin:
        sig = np.zeros( test_time.size, dtype=float )
        strt = 0
    else: 
        pn = np.random.uniform( -1*phase_noise, phase_noise )
        sig = pulse_waveform( test_time, pulse_width, bunch_shape=pulse_type, phase_shift=pn, phase_freq=rf, **kwargs )
        strt = 1
    
    # --- Build pulse train
    for n in range(strt, bunches):
        pn = np.random.uniform( -1*phase_noise, phase_noise )
        temp = pulse_waveform( test_time, pulse_width, bunch_shape=pulse_type, phase_shift=pn, phase_freq=rf, **kwargs)
        sig = np.concatenate( (sig, temp) )
    
    # --- Check for padding
    if pad_end:
        temp = np.zeros(test_time.size, dtype=float)
        sig = np.concatenate( (sig, temp) )    
    
    return np.linspace( 0.0, sig.size * dtt, sig.size ), sig

# -------------------------------- Other ------------------------------------- #

def split_pulse_1_to_4( 
        rf:List[float],      # -- Expecting three RF frequencies, can be initial RF freq only
        bunch_length:List[float],   # -- Expecting bunch lengths in seconds, can be single value for initial
        split_steps:int=1000,
        bunch_shape:str="cos", 
        pw_type:str="rms", 
        dt:float=1e-12,
        ):
    # --- Constants
    num_bunches = 4
    num_splits = 2

    # --- Create Time array
    if isinstance(rf, (list, tuple, np.ndarray)): 
        base_rf = min(rf)
        rf_group = rf
    else:  
        base_rf = rf
        rf_group = [base_rf, base_rf*2, base_rf*2]
    
    time_resolution = int( (2/(np.pi*base_rf)) // dt )
    time_resolution = max(time_resolution, 1000)

    test_time = np.linspace(-3/(2*np.pi*base_rf), 3/(2*np.pi*base_rf), time_resolution)

    # --- Phase steps
    dphi = 90/split_steps

    # --- Bunch lengths
    if isinstance(bunch_length, (list, tuple, np.ndarray)):
        init_bunch_length = bunch_length[0]
        split1_bl = bunch_length[1]
        split2_bl = bunch_length[2]
    else:
        init_bunch_length = bunch_length
        split1_bl = init_bunch_length/1.15
        split2_bl = split1_bl/1.15

    # --- Create bunches
    sig_test = []
    for i, f in enumerate(rf_group):
        for turns in range(split_steps):
            phase_shift = dphi*turns
            frame_signal = np.zeros(test_time.size,dtype=np.float64)

            # --- Break after 2 splits
            if i >= num_splits: break
            
            # --- Initial pulse
            if i == 0 and phase_shift < 45:
                init = pulse_waveform( test_time, init_bunch_length, bunch_shape=bunch_shape, pw_type=pw_type, phase_freq=f)
                frame_signal += init * _smooth_weight_on_phase(phase_shift, shift_quotient=45.0, max_weight=0.0, increase=False)

            # --- Splits 
            if i in [0,1]:
                centers_2 = [phase_shift, -1*phase_shift] if i == 0 else [180, -180]
                center_4 = [180+phase_shift,-180+phase_shift,180-phase_shift,-180-phase_shift] if i == 1 else [phase_shift, -1*phase_shift, phase_shift, -1*phase_shift]
                amp_factor = _smooth_weight_on_phase(phase_shift, shift_quotient=270.0,max_weight=1.167)
                amp_decay = 1.167*_smooth_weight_on_phase(phase_shift, shift_quotient=45.0, max_weight=0.0, increase=False)
                amplitude = amp_factor if i == 0 else amp_decay
                # --- Split 1
                for center in centers_2:
                    split = pulse_waveform( test_time, split1_bl, bunch_shape=bunch_shape, pw_type=pw_type, phase_freq=f, phase_shift=center, degrees=True )
                    frame_signal += split * amplitude
                # --- Split 2
                for center in center_4:
                    split = pulse_waveform( test_time, split2_bl, bunch_shape=bunch_shape, pw_type=pw_type, phase_freq=f, phase_shift=center, degrees=True )
                    frame_signal += split *_smooth_weight_on_phase(phase_shift)

            sig_test.append( frame_signal )
            
    return np.array(sig_test), test_time

def _smooth_weight_on_phase(phase_shift, shift_quotient=90.0, limit=45.0, max_weight=1.5, increase=True):
    """
    Calculate a weight factor based on phase shift and other parameters.

    Args:
        phase_shift (float): The phase shift value.
        shift_quotient (float, optional): The divisor used to scale the phase shift. Defaults to 90.0.
        limit (float, optional): The phase shift limit beyond which the weight is capped. Defaults to 45.0.
        max_weight (float, optional): The maximum weight value returned when the phase shift exceeds the limit. Defaults to 1.5.
        increase (bool, optional): Determines the direction of weight calculation. If True, weight increases with phase 
            shift; otherwise, it decreases. Defaults to True.

    Returns:
        float: The calculated weight factor based on the specified phase shift and parameters.
    """

    if increase:
        return (1 + phase_shift / shift_quotient) if phase_shift < limit else max_weight
    else:
        return (1 - phase_shift / shift_quotient) if phase_shift < limit else max_weight

# --- Helper function - select pulse waveform
def pulse_waveform( 
        test_time: np.ndarray,
        bunch_width: float, 
        bunch_shape: str="cos",
        **kwargs 
)->np.ndarray:
    """
    Generate a pulse waveform based on the given shape and parameters.

    Args:
        test_time (np.ndarray): Time array.
        bunch_shape (str, optional): Shape of the pulse. Options are "cos", "gauss", "skew_gauss", "doublet", "morlet". Defaults to "cos". 
        **kwargs: Additional keyword arguments for specific pulse shapes.

    Returns:
        np.ndarray: The generated pulse waveform.
    """
    sig = {
        "cos": cosine_square_pulse,
        "gauss": gaussian_pulse,
        "skew_gauss": skew_gaus_pulse,
        "doublet": gaus_doublet_pulse,
        "morlet": morlet_pulse,
    }
    if bunch_shape in sig.keys():
        return sig[bunch_shape]( test_time, bunch_width, **kwargs )
    else:
        raise ValueError(f"Invalid bunch shape: {bunch_shape}")

# --- Calculate RMS Pulse Width
def calculate_pulse_width(pulse_sig: np.ndarray, tp: float) -> float:
    """Calculate the pulse width of a given pulse signal.
    
    Args:
        pulse_sig (np.ndarray): Signal array of a single pulse.
        tp (float): Time period.
    
    Returns:
        float: Calculated RMS pulse width in seconds.
    """
    peaks, _ = sg.find_peaks(pulse_sig)
    half = sg.peak_widths(pulse_sig, peaks)
    pulse_width = (half[0] / (2 * np.sqrt(2 * np.log(2))))[0]
    return pulse_width * tp

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

def gamma_to_beta(gamma: Union[float, np.ndarray])->float:
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


