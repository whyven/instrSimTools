import numpy as np
import scipy.signal as sg
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def iir_filter(fc: float, order: int, typ: str, fs: float, ftype: str = "butter", check: bool = False) -> np.ndarray:
    """Create an IIR filter and optionally plot its frequency response.
    
    Args:
        fc (float): Cutoff frequency.
        order (int): Filter order.
        typ (str): Filter type (e.g., 'low', 'high', 'band').
        fs (float): Sampling frequency.
        ftype (str, optional): Filter design type. Defaults to "butter".
        check (bool, optional): Whether to plot the frequency response. Defaults to False.
    
    Returns:
        np.ndarray: IIR filter coefficients.
    """
    flt = sg.iirfilter(order, fc, analog=False, btype=typ, ftype=ftype, output='sos', fs=fs)
    w, h = sg.sosfreqz(flt)
    if check:
        # --- Check
        plt.subplot(2, 1, 1)
        db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        plt.plot(w * fs, db)
        plt.ylim(-75, 5)
        plt.grid(True)
        plt.yticks([0, -20, -40, -60])
        plt.ylabel('Gain [dB]')
        plt.title('Frequency Response')
        plt.subplot(2, 1, 2)
        plt.plot(w / np.pi, np.angle(h))
        plt.grid(True)
        plt.yticks([-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.ylabel('Phase [rad]')
        plt.xlabel('Normalized frequency (1.0 = Nyquist)')
        plt.show()
    return flt

def iir_filter_check(flt: np.ndarray, fs: float, frange: float | list[float]) -> int:
    """Check the frequency response of an IIR filter.
    
    Args:
        flt (np.ndarray): IIR filter coefficients.
        fs (float): Sampling frequency.
        frange (float | list[float]): Frequency range for the filter.
    
    Returns:
        int: Status code (0 for success).
    """
    # Generate an impulse response
    impres = np.zeros(3001)
    impres[501] = 1

    # apply the filter
    fimp = sg.sosfilt(flt, impres)

    # compute power spectrum
    fimpX = np.abs(fft(fimp)) ** 2
    hz = np.linspace(0, fs / 2, int(np.floor(len(impres) / 2) + 1))

    # plot
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(impres, 'k', label='Impulse')
    ax[0].plot(fimp, 'r', label='Filtered')
    ax[0].set_xlim([1, len(impres)])
    ax[0].set_ylim([-.06, .06])
    ax[0].legend()
    ax[0].set_xlabel('Time points (a.u.)')
    ax[0].set_title('Filtering an impulse')
    ax[0].grid(True)

    ax[1].plot(hz, fimpX[0:len(hz)], 'ks-')
    ax[1].grid(True)
    if not isinstance(frange, list):
        ax[1].plot([0, frange, frange, fs / 2], [1, 1, 0, 0], 'r')
    else:        
        ax[1].plot([0, frange[0], frange[0], frange[1], frange[1], fs / 2], [0, 0, 1, 1, 0, 0], 'r')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Attenuation')
    ax[1].set_title('Frequency response of filter (Butterworth)')
    
    plt.tight_layout()
    plt.show()
    return 0

def fir_filter(fcut: float | list[float], fs: float, ordr: int = None) -> np.ndarray:
    """Create an FIR filter.
    
    Args:
        fcut (float | list[float]): Cutoff frequency or frequencies.
        fs (float): Sampling frequency.
        ordr (int, optional): Filter order. Defaults to None.
    
    Returns:
        np.ndarray: FIR filter coefficients.
    """
    # filter parameters
    nyquist = fs / 2
    transw = .707
    if ordr is None:
        if isinstance(fcut, list):
            order = int(5 * fs / fcut[0])
        else:
            order = int(5 * fs / fcut)
    else:
        order = ordr

    # force odd order
    if order > 5000:
        order = 5000
    if order % 2 == 0:
        order += 1
    if isinstance(fcut, list):
        filt = sg.firwin(order, fcut, fs=fs, pass_zero=False)
    else: 
        filt = sg.firwin(order, fcut, fs=fs)
    return filt

def fir_filter_check(filtkern: np.ndarray, fs: float, fcut: float | list[float]) -> int:
    """Check the frequency response of an FIR filter.
    
    Args:
        filtkern (np.ndarray): FIR filter coefficients.
        fs (float): Sampling frequency.
        fcut (float | list[float]): Cutoff frequency or frequencies.
    
    Returns:
        int: Status code (0 for success).
    """
    fig, ax = plt.subplots(nrows=2, ncols=1)
    # time-domain filter kernel
    ax[0].plot(filtkern)
    ax[0].set_xlabel('Time points')
    ax[0].set_title('Filter kernel (firwin)')

    # compute the power spectrum of the filter kernel
    filtpow = np.abs(fft(filtkern)) ** 2
    # compute the frequencies vector and remove negative frequencies
    hz = np.linspace(0, fs / 2, int(np.floor(len(filtkern) / 2) + 1))
    filtpow = filtpow[0:len(hz)]

    # plot amplitude spectrum of the filter kernel
    ax[1].plot(hz, filtpow, 'ks-', label='Actual')
    if isinstance(fcut, list):
        ax[1].plot([0, fcut[0], fcut[0], fcut[1], fcut[1], fs / 2], [0, 0, 1, 1, 0, 0], 'ro-', label='Ideal')
        ax[1].set_xlim([0, fcut[1] * 4])
    else:
        ax[1].plot([0, fcut, fcut, fs / 2], [1, 1, 0, 0], 'ro-', label='Ideal')
        ax[1].set_xlim([0, fcut * 4])
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Filter gain')
    ax[1].legend()
    ax[1].set_title('Frequency response of filter (firwin)')
    
    plt.tight_layout()
    plt.show()
    return 0