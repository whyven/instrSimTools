import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple, List

# --- Scipy imports
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, leastsq
import scipy.signal as sg
import scipy.constants as const
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.special import erf

# --- Scikit-RF imports
import skrf as rf

# --- SPICE-Lab imports
from spicelib import SimRunner, AscEditor, RawRead
from spicelib.log.ltsteps import LTSpiceLogReader
from spicelib.simulators.ltspice_simulator import LTspice
from pathlib import Path


# --- Constants
XLABEL = "Frequency (Hz)"
YLABEL = "Attenuation (dB)"

# --- --- Attenuation equations
cable_atten = { "ldf1"		: (lambda x: (3.92700613e-01*np.sqrt(x)+1.22050618e-03*x+5.91674552e-05)/100 ), 
				"ldf2"		: (lambda x: (3.31327012e-01*np.sqrt(x)+1.07614300e-03*x-7.40976465e-06)/100 ), 
				"ldf4"		: (lambda x: (2.10628941e-01*np.sqrt(x)+6.23339147e-04*x-9.62924672e-05)/100 ), 
				"ldf4p5"	: (lambda x: (1.48449161e-01*np.sqrt(x)+6.88925267e-04*x-9.59345131e-05)/100 ), 
				"lmr240"	: (lambda x: (0.242080*np.sqrt(x)+0.00033*x)/30.48), # per Foot!!
				"lmr400"	: (lambda x: (0.122290*np.sqrt(x)+0.00026*x)/30.48 ),
				"sio2"		: (lambda x: (0.2923*np.sqrt(x)+0.00111*x)/30.48 ), 
}

# --- --- Attenuation equations
cable_velocity = { "ldf1"	: 0.86, 
				"ldf2"		: 0.85, 
				"ldf4"		: 0.88, 
				"ldf4p5"	: 0.88, 
				"fsj1kr"	: 0.82,
				"lmr240"	: 0.83,
				"lmr400"	: 0.84,
				"sio2"		: 0.80, 
	}

# ------------------- Cable Simulation ------------------- #

# --- Estimate Cable Attenuation --- #
def cable_atten_from_parameters(f: np.ndarray, z0: float, sigma: float, er: float, tand: float, r_cc: float, length: float):
    """Estimate the attenuation of a cable given its physical parameters.

    Args:
        f (np.ndarray): Frequency array.
        z0 (float): Characteristic impedance.
        sigma (float): Conductivity.
        er (float): Relative permittivity.
        tand (float): Loss tangent.
        r_cc (float): Conductor radius.
        lenght (float): Cable length.

    Returns:
        np.ndarray: Attenuation.
    """
    # --- Constants
    mu0 = const.mu_0
    eps0 = const.epsilon_0
    w = 2 * np.pi * r_cc
    
    a1 = (length / (2*w*z0) ) * np.sqrt( np.pi * mu0/sigma ) # Skin effect
    a2 = (length*np.pi*tand*np.sqrt(er))/const.c # Dielectric loss

    # --- Attenuation vs frequency
    alpha = np.exp(-1*(a1*np.sqrt(f) + a2*f))

    return alpha

# --- Function for determining cable propagation delay
def calc_delay( cable, length, velocity=None ):
	m_per_ft = 0.3048
	if (velocity is None):
		return ( ( float(length) * (m_per_ft / const.c) ) / cable_velocity.get(cable, 0.85) ) * 1.0e9
	else:
		return ( ( float(length) * (m_per_ft / const.c) ) / float(velocity) ) * 1.0e9

# ------------------- Pole-Zero Approximation ----------------- #

# --- Functions
def pz( x, p, z):
    return (1+((2*np.pi*x)/(2*np.pi*z))**2)/(1+((2*np.pi*x)/(2*np.pi*p))**2)

# -- Pole Equation
def p(x,p):
    return 1/(1+((2*np.pi*x)/(2*np.pi*p))**2)


# --- 6-pole,5-zero
def pz_fit_six(init,x):
    return np.sqrt( 
        pz(x,init[0],init[1])*pz(x,init[2],init[3])*pz(x,init[4],init[6])*pz(x,init[6],init[7])*pz(x,init[8],init[9])*p(x,init[10])
    )

def pz_fit6(x, a,b,c,d,e,f,g,h,i,j,k):
    return np.sqrt( 
        pz(x,a,b)*pz(x,c,d)*pz(x,e,f)*pz(x,g,h)*pz(x,i,j)*p(x,k)
    )

# --- Function that does r/c calculation
def pz_to_rc(impedance, pz_ary):
    """ Converts the pole/zero values into resistor/capacitor values.
        
        pz_ary - Expecting to be an array/list of float values with length = 10 and
                 the structure of pole1,zero1,...,poleN,zeroN
        
        returns - Two lists. An array of ref designators. An array of float values. 
                  The structure r1,c1,...,rN,cN
    """
    ref_des = []
    rc_list = []
    
    # -- Seperate pole/zero info
    p_val = pz_ary[[0,2,4,6,8,10]]
    z_val = pz_ary[[1,3,5,7,9]]
    
    ## -- leastsq
    for pos,pv,zv in itertools.zip_longest(np.arange(len(p_val)),p_val,z_val):
        if zv is None:
            ref_des.append("C{}".format(pos+1))
            rc_list.append( (1/(2.*np.pi*impedance*pv)) )
        else:
            ref_des.append("R{}".format(pos+1))
            rc_list.append( (impedance*((zv/pv)-1)) )
            
            ref_des.append("C{}".format(pos+1))
            rc_list.append( (1/(2.*np.pi*impedance*zv)) )
    
    return ref_des, rc_list
	
# --- Function to convert from dB to V/V
def db_to_vv( data, f_range, cable_lengths ):
	"""
	"""
	temp_df = pd.DataFrame()
	temp_fr = np.linspace(f_range[0],f_range[-1],1000)
	# --- If given a dataframe, cycle through lengths 
	if isinstance( data, pd.DataFrame):
		for l in cable_lengths:
			# --- --- convert to V/V
			vv = np.array([10**(-x/20) for x in data[l]])
			# --- --- create an interpolation function
			temp_func = interp1d(f_range, vv, kind="cubic")
			vv_n = temp_func(temp_fr)
			temp_df[l] = vv_n.tolist()
	else:
		vv = np.array([10**(-x/20) for x in data])
		# --- --- create an interpolation function
		temp_func = interp1d(f_range, vv, kind="cubic")
		vv_n = temp_func(temp_fr)
		temp_df = vv_n.tolist()
	
	return temp_df, temp_fr*1e6

# --- Function for cycling through regression fit
def pz_reg_fit( cable_atten, f_range, init, length=100, 
            slicer=slice(None,None,None), bounds=(1000,np.inf), 
            max_attempts=5000, *args, **kwargs ):
	"""
	"""
	if isinstance(cable_atten, pd.DataFrame):
		vals = []
		rsq = []
		for l in cable_atten.columns[slicer]:
			popt, pcov = curve_fit(pz_fit6, f_range, cable_atten[l], init, bounds=bounds, max_nfev=max_attempts)
			rsq.append( corcoef( pz_fit6, popt, f_range, cable_atten[l]) )
			vals.append(popt)
		return vals, rsq
		
	else:
		try:
			popt, pcov = curve_fit(pz_fit6, np.array(f_range), np.array(cable_atten), init, bounds=bounds, max_nfev=max_attempts)
		except RuntimeError:
			try:
				popt, pcov = curve_fit(pz_fit6, np.array(f_range), np.array(cable_atten), init, bounds=(10,np.inf), max_nfev=10000 )
			except RuntimeError:
				raise
		rsq = corcoef( pz_fit6, popt, np.array(f_range), np.array(cable_atten) )
		return popt, rsq

# --- Function for determining poles and zeros from 
def conv_to_rc( cable_len, data, impedance=50.0 ):
	# --- Iterate through the fit values for poles/zeros and calc r/c values
	rc_vals = pd.DataFrame()
	if isinstance(data, pd.DataFrame):
		for clabel, vals in zip(cable_len, data):
			ref_des, rc_list = pz_to_rc( impedance, vals )
			rc_vals["SPICE Ref Des"] = ref_des
			rc_vals["{}m".format(clabel)]= rc_list
			
	else:
		ref_des, rc_list = pz_to_rc( impedance, data )
		rc_vals.index = ref_des
		rc_vals["{}m".format(cable_len)]= rc_list
	return rc_vals

# --- Function for generating subcircuit
def create_subcircuit( rc_vals, cable, length, impedance=50.0, velocity=0.85 ):
	
	d = rc_vals["{length}m".format(length=length)]
	d["cable"] = str(cable)
	d["length"] = int(float(length))
	delay = calc_delay(cable, length, velocity)
	d["delay"] = "{:0.3f}n".format(delay)
	d["Z"] = impedance
	d["date"] = str(date.today())
	
	# --- Open up the generic subcircuit file 
	with open( "./static/Generic_subcircuit.lib","r") as f:
		fl = f.read()
	nfl = fl.format(**d)
	
	return nfl

# ------------------- Statistics ------------------- #

# --- Regression Functions 
def regression(xi, yi, reg_type=None, init=None,):
	xi, yi = check_length( xi, yi )
	
	if xi is None:
		return None, None, None
	xa = np.array(xi)
	if (xa[-1] >= 1e9 ):
		xa *= 1e-6
	ya = np.array(yi)

	# --- Define dictionary of initial conditions for regression types
	init_dict = {"pow1"     :   (1,1,1),
				 "pow2"     :   (1,1,1,1,1), 
				 "sqrt"      :  (1,1,1),
				 }

	# --- Define dictionary of functions for regression types
	func_dict = { "pow1"    :   lambda tpl,x : tpl[0] + tpl[1]*x**tpl[2],
				  "pow2"    :   lambda tpl,x : tpl[0] *(tpl[1] + tpl[2]*(x+tpl[3])**tpl[4]),
				  "sqrt"     :  lambda tpl, x : tpl[0] *np.sqrt(x)+tpl[1]*x + tpl[2],
				  }

	# --- Configure functions list
	# --- --- Make a list of requested functions
	if reg_type is None:                    # -- All regression tests
		funcs = func_dict.keys()
	elif not isinstance(reg_type,list):  # -- One regression tests
		funcs = [reg_type]
		inits = [init]
	else:                                   # -- Multiple regression tests
		funcs = reg_type
		inits = init
	# --- --- Update initial conditions based on input 	   
	if init is not None:
		for init_key in range(len(funcs)):
			init_dict[funcs[init_key]] = inits[init_key]

	# --- Run regression test
	finalFunc = {}
	rsq = {}
	for key in funcs:
		ErrorFunc= lambda tpl, x, y: func_dict[key](tpl,x)-y
		tplfinal, success = leastsq(ErrorFunc, init_dict[key][:], args=(xa,ya))
		finalFunc[key] = tplfinal
		rsq[key]= corcoef(func_dict[key], tplfinal, xa, ya, mode="REG" ) 

	return {k:func_dict[k] for k in func_dict.keys() if k in funcs}, finalFunc, rsq

# --- Function for determining the correlation coefficient
def corcoef(func, tpl, x, y, mode="PZ"):
	# --- Check mode
	if mode == "PZ":
		y_hat = func(x, *tpl)
	else:
		y_hat = func(tpl,x)
	y_bar = y.sum() / len(y)
	ssreg = np.sum((y_hat - y_bar)**2)
	sstot = np.sum((y - y_bar)**2)
	ssres = np.sum((y-y_hat)**2)
	
	return 1 - ssres / sstot

def reg_plot( x, y, func, func_vals, rsq, cable_name, xlbl=XLABEL, ylbl=YLABEL, ):
	fig = Figure()
	ax = fig.add_subplot(111)
	
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

# ------------------- Import Function ------------------- #

# --- Import S-Parameters
def import_spram( path=None ):
    if path is None:
        path_t = input('file path: ')
        name = input('file name (include extension): ')
        ntwk_path = path_t + '\\' + name
    else:
        ntwk_path = path

    return rf.Network( ntwk_path )

# --- Import Attenuation Data
def import_atten( path=None ):
    if path is None:
        path_t = input('file path: ')
        name = input('file name (include extension): ')
        atten_path = path_t + '\\' + name
    else:
        atten_path = path

    return pd.read_csv( atten_path )

# ------------------- Other ------------------- #

# --- Check if a number
def is_num(x):
	"""
	Check if x is a number (i.e. can be converted to a float)

	Parameters
	----------
	x : object
		the object to check

	Returns
	-------
	True if x is a number, False otherwise
	"""
	try:
		float(x)
		return True
	except ValueError:
		return False

# --- Compare lengths of two arrays
def check_length( x_data, y_data ):
	min_len = min( x_data.size, y_data.size )
	if min_len == 0:
		return None, None
	else:
		return x_data[:min_len], y_data[:min_len]

def convert_lengths(values, from_unit, to_unit: str, show: bool = False):
    """
    Convert lengths between imperial and metric units.

    Args:
        values (float, list, or tuple): Length or list/tuple of lengths to convert.
        from_unit (str or list/tuple): Unit(s) to convert from. Can be a single unit or a list/tuple matching the length of values.
        to_unit (str): Unit to convert to. Supported: 'mm', 'cm', 'm', 'in', 'ft', 'yd'
        show (bool): If True, prints each conversion.

    Returns:
        float or list: Converted length(s).
    """

    # Conversion factors to meters
    to_meters = {
        'mm': 0.001,
        'cm': 0.01,
        'm': 1.0,
        'in': 0.0254,
        'ft': 0.3048,
        'yd': 0.9144
    }

    # Normalize inputs
    is_single = False
    if isinstance(values, (int, float)):
        values = [values]
        is_single = True
    elif isinstance(values, (list, tuple)):
        values = list(values)
    else:
        raise TypeError("values must be a number, list, or tuple.")

    # Normalize from_unit
    if isinstance(from_unit, str):
        from_units = [from_unit.lower()] * len(values)
    elif isinstance(from_unit, (list, tuple)):
        if len(from_unit) != len(values):
            raise ValueError("Length of from_unit list must match length of values.")
        from_units = [u.lower() for u in from_unit]
    else:
        raise TypeError("from_unit must be a string or list/tuple of strings.")

    # Normalize to_unit
    to_unit = to_unit.lower()

    # Validate units
    for u in from_units:
        if u not in to_meters:
            raise ValueError(f"Unsupported from_unit: {u}")
    if to_unit not in to_meters:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    # Convert
    results = []
    for val, f_unit in zip(values, from_units):
        meters = val * to_meters[f_unit]
        converted = meters / to_meters[to_unit]
        results.append(converted)

        if show:
            print(f"{val} {f_unit} = {converted:.4f} {to_unit}")

    return results[0] if is_single else results


# --- Create a WCM Circuit
def simple_wcm_circuit( Cb, Cc, Cp, Ls, Lby, Rp, *args, **kwargs ):
	# --- Initialize the circuit frequency domain
	freq = rf.Frequency( start=0.0001, stop=10000, unit="MHz", npoints=50001, sweep_type='log' )
	wcm_media = rf.DefinedGammaZ0( freq, z0=50, gamma=1j*freq.w/rf.c )
	
	# --- Define the components
	Cbr = wcm_media.capacitor( Cb, name="Cbr") 			# Ceramic Break Capacitance
	Ccc = wcm_media.capacitor( Cc, name="Ccc")	        # Cocentric Cylinders Capacitance -- not sure about this one..
	EPC = wcm_media.capacitor( Cp, name="EPC")			# Equivalent Parallel Capacitance
	RLs = wcm_media.inductor( Ls, name="RLs" )			# Series Inductance
	# RLp = wcm_media.inductor( 41.7e-3, name="RLp" )			# Parallel Inductance
	Llf = wcm_media.inductor( Lby, name="Llf" )			# Low frequency Inductance -- Current bypass assembly
	RWCM = wcm_media.resistor( Rp, name="RWCM" )		# Equivalent parallel resistance of WCM
	gnd = rf.Circuit.Ground( freq, name="gnd" )
	port1 = rf.Circuit.Port( freq, name="port1", z0=50 )
	port2 = rf.Circuit.Port( freq, name="port2", z0=50 )

	# --- Make Netlist
	cnx = [
    [(port1,0),(Llf,0),(Cbr,0),(Ccc,0), (EPC,0), (RLs,0), (port2,0)], # (RLp,0),(port2,0)],
    [(RLs,1), (RWCM,0),],
    [(gnd,0),(Llf,1),(Cbr,1),(Ccc,1), (EPC,1), (RWCM,1),] # (RLp,1),]
	]

	cir = rf.Circuit(cnx)
	ntw = cir.network
	
	return ntw

# --- Calculate Parallel Plate Capacitance
def calc_cap_parallel_plate( ep_r, A, d ):
	return ep_r * const.epsilon_0 * A / d

# --- Calculate Capacity of Cocentric Cylinders
def calc_cap_cocentric_cylinders( ep_r, d_outer, d_inner, length ):
	return ( ( ep_r * const.epsilon_0 * 2 * np.pi ) / np.log( d_outer / d_inner ) ) * length

# --- Calculate Inductance of Cocentric Cylinders
def calc_ind_cocentric_cylinders( mu_r, d_outer, d_inner, length ):
	return ((mu_r * const.mu_0) / (2*np.pi)) * np.log( d_outer / d_inner ) * length

# --- Calculate series inductance and parallel capacitance from S11
def calc_resistor_parasitics_from_s11( ntwk ):
	# Convert to Z-parameters (input impedance)
	z = ntwk.z[:, 0, 0]  # Get Z11 over all frequencies
	f = ntwk.f  # Frequency array in Hz

	# Calculate series inductance from imaginary part of Z
	X = np.imag(z)
	L_series = X / (2 * np.pi * f)

	# Convert Z to admittance (for parallel capacitance extraction)
	y = 1 / z
	B = np.imag(y)
	C_parallel = -B / (2 * np.pi * f)

	# Calculate parallel inductance if B is negative
	L_parallel = -1 / (2 * np.pi * f * B)

	return L_series, C_parallel, L_parallel, f

# --- Calculate 3dB bandwidth
def measure_3db_bandwidth( ntwk ):
	freq = ntwk.f
	magnitude_db = ntwk.s21_db.to_dataframe().values[:,0]
	
	max_db = np.max(magnitude_db)
	threshold = max_db - 3
	
	# Find where the magnitude crosses the -3 dB line
	indices = np.where(magnitude_db >= threshold)[0]

	if len(indices) < 2:
		return 0.0  # Cannot determine bandwidth

	f_low = freq[indices[0]]
	f_high = freq[indices[-1]]
	return f_high - f_low, f_low, f_high, max_db

# -----------
def prepare_output_folder(path: str, folder_name: str = "batch_output") -> Path:
    """
    Create and return the output folder path for simulation results.

    Args:
        path (str): Base directory path where the folder will be created.
        folder_name (str, optional): Name of the output subdirectory. Defaults to 'batch_output'.

    Returns:
        Path: Full path to the created (or existing) output folder.
    """
    output_path = Path(path) / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def setup_simulation(
    asc_path: str,
    output_path: Union[str, Path],
    default_params: Dict[str, float],
    instructions: List[str] = None
) -> Tuple[SimRunner, AscEditor]:
    """
    Initialize and configure the LTspice simulation with default parameters.

    Args:
        asc_path (str): Path to the .asc LTSpice schematic file.
        output_path (Union[str, Path]): Folder where output will be stored.
        default_params (Dict[str, float]): Dictionary of component parameters to set in the circuit.
        instructions (List[str], optional): Additional SPICE instructions to add to the simulation.

    Returns:
        Tuple[SimRunner, AscEditor]: Configured SimRunner and AscEditor objects.
    """
    runner = SimRunner(simulator=LTspice, output_folder=output_path)
    editor = AscEditor(asc_path)

    editor.set_parameters(**default_params)
    if instructions:
        editor.add_instructions(*instructions)
    editor.set_parameter('run', 0)
    return runner, editor

def generate_waveforms(
    time_ary: np.ndarray,
    signal_dict: Dict[str, np.ndarray],
    output_dir: str
) -> Dict[str, str]:
    """
    Export a dictionary of waveforms to PWL files for SPICE simulation.

    Args:
        time_ary (np.ndarray): Time array corresponding to the waveforms.
        signal_dict (Dict[str, np.ndarray]): Dictionary of waveforms keyed by name.
        output_dir (str): Path where waveform files should be saved.

    Returns:
        Dict[str, str]: Dictionary mapping waveform names to saved file paths.
    """
    from spicelib.helpers import export_to_spice_pwl  # Local import for compatibility

    wave_paths = {}
    spice_time = time_ary - time_ary[0]

    for key, waveform in signal_dict.items():
        file_path = os.path.join(output_dir, f"Waveform_{key}.txt")
        export_to_spice_pwl(spice_time, waveform, file_path)
        wave_paths[key] = file_path

    return wave_paths

def run_batch_simulations(
    runner: SimRunner,
    editor: AscEditor,
    waveforms: Dict[str, str],
    use_alt_solver: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run batch LTSpice simulations with each input waveform and collect results.

    Args:
        runner (SimRunner): SimRunner object for executing simulations.
        editor (AscEditor): AscEditor object used to modify and control the simulation.
        waveforms (Dict[str, str]): Dictionary mapping waveform names to PWL file paths.
        use_alt_solver (bool, optional): Whether to use the alternate SPICE solver (-alt). Defaults to True.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
            Dictionary keyed by waveform name with values being tuples of (time, V(wcm_in), V(wcm_out)).
    """
    results = {}

    # Ensure consistent transient setup
    editor.remove_Xinstruction(r"\.TRAN.*")
    editor.add_instructions(".TRAN 1p 100n 0 10p")

    for key, waveform_path in waveforms.items():
        print(f"Simulating response to: {key}")
        editor['I1']['Value'] = f"PWL SCOPEDATA={waveform_path}"
        opts = ['-alt'] if use_alt_solver else ['-norm']
        runner.run(editor, switches=opts, exe_log=True)

    # Read results
    for raw, log in runner:
        print(f"Raw file: {raw}, Log file: {log}")
        raw_data = RawRead(raw)
        x = raw_data.get_wave('time')
        y1 = raw_data.get_wave('V(wcm_in)')
        y2 = raw_data.get_wave('V(wcm_out)')
        waveform_id = str(raw)[-1]  # adjust this if needed for unique names
        results[waveform_id] = (x, y1, y2)

        # Optional: parse log data
        log_data = LTSpiceLogReader(log)

    print(f"Successful/Total Simulations: {runner.okSim}/{runner.runno}")
    return results
