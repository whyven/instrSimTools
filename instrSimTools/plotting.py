import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.constants as const
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Union, Dict, Tuple, List, Optional, Any

# --- Constants
# Utility for time unit scaling
TIME_UNIT_SCALE = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}
FREQ_UNIT_SCALE = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

# ------------------- Plotting functions from Waveforms ------------------- #

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
def waterfall_plot(
    waveforms: list[np.ndarray],
    time_array: np.ndarray,
    num_waveforms_to_plot: int = 50,
    amplitude_scaling: int = 5,
    title: str = "Waterfall Plot of Waveforms",
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    xlabel: str = "Time (Samples)",
    ylabel: str = "Amplitude + Offset",
    figsize: tuple[int, int] = (10, 6),
    **kwargs
) -> None:
    """
    Creates a waterfall plot of a subset of waveforms from a list.

    Args:
        waveforms: A list of numpy arrays, where each array represents a waveform.
        time_array: A numpy array representing the time values for the waveforms.
        num_waveforms_to_plot: The number of evenly spaced waveforms to display.
        title: The title of the plot.
        xlim: The x-axis limits (tuple of two floats).
        ylim: The y-axis limits (tuple of two floats).
        xlabel: The x-axis label.
        ylabel: The y-axis label.
        figsize: The figure size (tuple of two ints).

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

    plt.figure(figsize=figsize)

    for i, waveform in enumerate(selected_waveforms):
        plt.plot(time_array, waveform + i * amplitude_scaling, linewidth=0.75, color='darkblue', **kwargs) # Add offset to each waveform

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def animate_bunch_splitting(
    waveforms: list[np.ndarray],
    time_array: np.ndarray,
    title: str = "Waterfall Plot of Waveforms",
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    xlabel: str = "Samples",
    ylabel: str = "Amplitude |a.u.|",
    save_file: str = None,
    **kwargs
) -> None:
    """
    Animate a set of waveforms by plotting each waveform in sequence.

    Parameters
    ----------
    waveforms : list of numpy arrays
        The list of waveforms to animate.
    time_array : numpy array
        The time array to use for the x-axis of the plot.
    title : str, optional
        The title of the plot.
    xlim : tuple of two floats, optional
        The x-axis limits.
    ylim : tuple of two floats, optional
        The y-axis limits.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    save_file : str, optional
        The file path to save the animation to. If not provided, the animation is displayed but not saved.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    line, = ax.plot(time_array, waveforms[0], **kwargs)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def update(frame):
        line.set_ydata(waveforms[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(waveforms), interval=50, blit=True)
    if save_file:
        ani.save(save_file, writer="pillow", fps=20)

    plt.show()

# ------------------- Plotting functions from circuit ------------------- #

# --- Regression Fit Plots
def cableAtten_reg_plot( x, y, func, func_vals, rsq, cable_name, xlbl="Frequency (MHz)", ylbl="Attenuation (dB)", ):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), dpi=100) # fig = plt.Figure()
	# ax = fig.add_subplot(111)
	
	xa, ya = check_length(x, y)
	if (xa[-1] >= 1e9 ):
		xa *= 1e-6
	# --- Plotting
	xx1 = np.linspace(xa.min(), xa.max(), len(xa)*20)
	ax.plot(xa,ya,'bo')

	for key in func:
		ax.plot(xx1, func[key](func_vals[key], xx1), '--')

	# --- --- Set up the legend label
	leg = ['data']
	for key, val in rsq.items():
		leg.append( key+', $R^2$ = {:.4f}'.format(val)  )        
		ax.legend( leg )

	# --- --- Set up the axis'
	ax.set_xlabel(xlbl)
	ax.set_ylabel(ylbl)
	ax.grid('on')
	fig.suptitle('{} Attenuation Over Frequency w/ Fit'.format(cable_name))
	# fig.tight_layout()
	
	return fig

# --- Plot Batch Simulation Results
def plot_spice_batch_results(
    batch_data: Dict[str, Tuple[str, Dict[str,np.ndarray]]],
    signals: List[str],
    monitors: List[str],
    time_units: str = 'ns',
    title: str = "SPICE Batch Simulation Results",
    xlim: List[float] = None,
    xlabel: str = "Time [{}]",
    ylim: List[float] = None,
    ylabel: Union[str, List[str]] = "Voltage (V)",
    dual_y: List[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Plot input/output waveforms from a batch of SPICE simulations.

    Args:
        batch_data (Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]): 
            Dictionary of batch results with keys as run names and values as tuples:
            (time_array, input_waveform, output_waveform).
        time_units (str, optional): Units for time axis label. Defaults to 'ns'.
        title (str, optional): Plot title. Defaults to "SPICE Batch Simulation Results".
        figsize (Tuple[int, int], optional): Size of the plot figure. Defaults to (10, 5).

    Returns:
        None
    """
    time_factor = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1}
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    
    # --- Setup plot for dual y axis
    if dual_y:
        color = ['coral','salmon','tomato','orangered','red','crimson','brown']
        ax1 = ax.twinx()
        if isinstance(ylabel, list):
            ax.set_ylabel(ylabel[0])
            ax1.set_ylabel(ylabel[1], color=color[0])
            ax1.tick_params(axis='y', labelcolor=color[0])
    
    for i, (run_name, mons) in enumerate(batch_data.items()):
        if run_name in signals:
            time, mon_dict = mons
            for key, val in mon_dict.items():
                if key in monitors:
                    if dual_y and key in dual_y:
                        ax1.plot( time * time_factor[time_units], val, 
                                 label=f'{run_name} {key}', linestyle='--', 
                                 alpha=0.7, color=color[i%len(color)] )
                    else:
                        ax.plot(time * time_factor[time_units], val, 
                                label=f'{run_name} {key}', linestyle='-', alpha=0.8, )
    
    # --- Formatting plot
    h1, l1 = ax.get_legend_handles_labels()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if dual_y:
        ax1.set_ylim(ax.get_ylim())
        h2, l2 = ax1.get_legend_handles_labels()
    ax.set_xlabel(xlabel.format(time_units))
    ax.set_title(title)
    ax.legend(handles=(h1+h2), labels=(l1+l2), loc='best', fontsize=8)
    ax.grid(True, which='both', ls='--', lw=0.5, color='gray')
    plt.tight_layout()
    plt.show(fig)

    return fig

# --- Plot Single Simulation Results
def plot_spice_results(
    data: Union[Tuple[np.ndarray], np.ndarray],
	x_array: np.ndarray,
    x_units: str = 'ns',
    title: str = "SPICE Simulation Results",
    xlabel: str = "Time (ns)",
    ylabel: str = "Voltage (V)",
    legend: Union[List[str], str] = None,
    xlim_ns: Tuple[float, float] = None,
    ylim_ns: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (10, 5),
	**kwargs
) -> None:
    
    time_factor = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1}
    plt.figure(figsize=figsize)

    # --- Plot input/output waveforms
    if isinstance(data, tuple):
        for d in data:
            plt.plot(x_array * time_factor[x_units], d, **kwargs)
    else:
        plt.plot(x_array * time_factor[x_units], data, **kwargs)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', ls='--', lw=0.5, color='gray')

    if legend:
        plt.legend(legend, loc='best', fontsize=8)
    if xlim_ns:
        plt.xlim(xlim_ns)
    if ylim_ns:
        plt.ylim(ylim_ns)
    
    plt.tight_layout()
    plt.show()

# ------------------- Basic Plotting functions ------------------- #

# ---- Single Comparison Plot
def compare_plot_single(
    time: Union[List[float], pd.Series],
    ideal_wfm: Union[List[float], pd.Series],
    wfms: Dict[str, pd.DataFrame],
    time_unit: str = "ns",
    ylim: List[float] = None,
    xlim: List[float] = None,
    title: str = "Comparison Plot: Ideal vs. Filtered Waveform",
    xlabel: str = None,
    ylabel: str = "Amplitude (a.u.)",
    **kwargs,
):
    scale = TIME_UNIT_SCALE.get(time_unit, 1.0)
    fig, ax = plt.subplots()

    ax.plot(time * scale, ideal_wfm, label="ideal", **kwargs)

    for label, df in wfms.items():
        ax.plot(df["Time"] * scale, df[label], label=label, **kwargs)

    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.show()

    return fig

# --- Dual Comparison Plot
def compare_plot_multiple(
    time: Union[List[float], pd.Series],
    ideal_wfm: Union[List[float], pd.Series],
    wfms: Dict[str, pd.DataFrame],
    split_keys: List[str],
    n_cols: int ,
    n_rows: int,  # "row" or "col"
    time_unit: str = "ns",
    ylim: List[float] = None,
    xlim: List[float] = None,
    titles: List[str] = None,
    sup_title: str = "Comparison Plot: Ideal vs. Filtered Waveform",
    xlabel: str = None,
    ylabel: str = "Amplitude (a.u.)",
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
):
    scale = TIME_UNIT_SCALE.get(time_unit, 1.0)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, key in enumerate(split_keys[:len(axes)]):
        ax = axes[i]
        ax.plot(time * scale, ideal_wfm, label="ideal", **kwargs)

        for label, df in wfms.items():
            if key in label:
                ax.plot(df["Time"] * scale, df[label], label=label, **kwargs)

        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(ylabel)
        ax.set_title(titles[i] if titles else f"{key}")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True)
        ax.legend()

    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show(fig)

    return fig

# --- Dual y-axis comparison plots - Row seperated
#FIXME : Single plot with dual y-axes not working... crashing
def dual_axis_comparison_plot(
    time: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
    y1_data: Union[List[np.ndarray], Tuple[np.ndarray, ...]],
    y2_data: Union[List[np.ndarray], Tuple[np.ndarray, ...]],
    timescale_factor: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
    xlabel: str = "Time",
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: str = "Primary Y-Axis",
    y2label: str = "Secondary Y-Axis",
    plot_titles: Optional[List[str]] = None,
    sup_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create side-by-side comparison plots with dual y-axes.
    Supports shared or per-subplot time arrays.

    Args:
        time: Either a single np.ndarray for all subplots, or a list/tuple of np.ndarrays (one per subplot).
        y1_data: List or tuple of np.ndarrays for primary axis data.
        y2_data: List or tuple of np.ndarrays for secondary axis data.
        timescale_factor: Scale applied to time axis (e.g., 1e9 for ns).
        xlim: Optional x-axis limits (min, max) after scaling.
        xlabel: Label for x-axis.
        ylim: Optional y-axis limits for primary axis.
        ylabel: Label for primary y-axis.
        y2label: Label for secondary y-axis.
        plot_titles: Optional list of subplot titles (length matches y1_data).
        sup_title: Optional super title for the figure.
        figsize: Figure size.
    """
    num_plots = len(y1_data)
    if len(y2_data) != num_plots:
        raise ValueError("y1_data and y2_data must have the same length.")

    # Convert time to per-subplot list if it's a single array
    if isinstance(time, (np.ndarray, list, tuple)):
        if isinstance(time, np.ndarray):
            time_list = [time] * num_plots
        else:
            if len(time) != num_plots:
                raise ValueError("Length of time list/tuple must match number of plots.")
            time_list = list(time)
    else:
        raise TypeError("time must be a numpy array or a list/tuple of numpy arrays.")

    fig, axes = plt.subplots(nrows=1, ncols=num_plots, sharey=True, figsize=figsize)
    if num_plots == 1:
        axes = [axes]  # make iterable

    twin_axes = [ax.twinx() for ax in axes]

    for idx, (ax, ax2, t_arr, y1, y2) in enumerate(zip(axes, twin_axes, time_list, y1_data, y2_data)):
        ax.plot(t_arr * timescale_factor, y1, label=ylabel)
        ax2.plot(t_arr * timescale_factor, y2, ":", color="green", label=y2label)

        if ylim:
            ax.set_ylim(ylim)
            ax2.set_ylim(ylim)
        else:
            ax2.set_ylim(ax.get_ylim())

        ax.grid(True, which="both", axis="both")
        ax2.grid(True, which="both", axis="both", linestyle=":", alpha=0.8)
        ax2.tick_params(axis='y', labelcolor="green")
        ax.set_xlabel(xlabel)

        if xlim:
            ax.set_xlim(xlim)

        if plot_titles and idx < len(plot_titles):
            ax.set_title(plot_titles[idx])

        if idx == 0:
            ax.set_ylabel(ylabel)
        if idx == num_plots - 1:
            ax2.set_ylabel(y2label, color="green")

    # Hide redundant secondary y-axis labels
    for ax2 in twin_axes[:-1]:
        ax2.yaxis.set_tick_params(labelright=False)

    if sup_title:
        plt.suptitle(sup_title)

    fig.tight_layout()
    plt.show(fig)

    return fig

# ----------- Helper functions

# --- Check_length
def check_length( x_data, y_data ):
	min_len = min( x_data.size, y_data.size )
	if min_len == 0:
		return None, None
	else:
		return x_data[:min_len], y_data[:min_len]
    
