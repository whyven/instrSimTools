import itertools
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, leastsq
import scipy.signal as sg
import scipy.constants as const
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.special import erf
import matplotlib.pyplot as plt
import skrf as rf

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
	try:
		temp = float(x)
		return True
	except Exception as e:
		return False

# --- Compare lengths of two arrays
def check_length( x_data, y_data ):
	min_len = min( x_data.size, y_data.size )
	if min_len == 0:
		return None, None
	else:
		return x_data[:min_len], y_data[:min_len]