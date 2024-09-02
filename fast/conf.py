'''
Needs to read in python/YAML file and spit out a dictionary, taking into account 
default values for certain parameters.
'''
import importlib.util
import numpy
import logging
from collections import UserDict
import copy

logger = logging.getLogger(__name__)

class ConfigParser():
    """
    Config file parser for FAST. 

    Parameters:
        arg (str or FastConfig): Either a file name of the config file, or a previously 
            created FastConfig object, or a dict with config values
    """
    def __init__(self, arg):
        
        self.defaults = []
        self.config = FastConfig()
        self.init_params()

        if isinstance(arg, (FastConfig, dict)):
            for key in arg:
                self.config[key] = arg[key]
            self.fname = None
        elif type(arg) == str:
            self.fname = arg
            self.load(arg)
        else:
            raise Exception("Either config file name or params dict required")
                
    def load(self, fname):
        '''
        Load config file into dictionary

        Parameters:
            fname (string): config file location
        '''

        if fname.split('.')[-1] == "py":
            # python config file, import from filename
            spec = importlib.util.spec_from_file_location("", fname)
            conf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf_module)
            
            for key in conf_module.p:
                self.config[key] = conf_module.p[key]

            return

        raise Exception("Require .py config file")

    def init_params(self):
        '''
        Set defualt values for all config parameters from global PARAMS list
        '''
        self.defaults = PARAMS
        for d in self.defaults:
            self.config[d.name] = d


class FastConfig(UserDict):
    '''
    Wrapper around a dict of FastParam parameter values. 
    '''
    def __getitem__(self, key):
        return self.data[key].value

    def __setitem__(self, key, value):
        try:
            self.data[key].value = value
        except KeyError:
            self.data.__setitem__(key, value)

    def copy(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        return self.data.__repr__()

class FastParam():
    """
    FAST config parameter class. 

    Parameters:
        name (str): name of the parameter
        value: (initial) value for the parameter
        category (str): category of the parameter. For compatibility with future 
            versions where I organise the config files a bit better.
        bounds (tuple): (min, max) allowed values for the parameter. None indicated 
            no bound 
        allowed_values (list): allowed non-numeric values (e.g. ["opt", None])
        required_len (int or None): required length if an array parameter.

    """
    def __init__(self, name, value, category, bounds=[None, None], 
                 allowed_values=[], required_len=None, unit=None, info=""):

        self._bounds = bounds 
        if self._bounds[0] == None: self._bounds[0] = -numpy.inf 
        if self._bounds[1] == None: self._bounds[1] = +numpy.inf

        self._allowed_values = allowed_values
        self._category = category
        self._required_len = required_len

        self.name = name
        self.value = value
        self.unit = unit 
        self.info = info

    @property
    def value(self):
        return self._value 

    @value.setter
    def value(self, v):
        self.check(v)
        self._value = v

    def check(self, v):
        
        if isinstance(v, (numpy.ndarray, list, tuple)):
            v = numpy.array(v)
            if not ((v >= self._bounds[0]).all() and (v <= self._bounds[1]).all()):
                raise ValueError(f"One or more values in parameter {self.name} out of bounds {self._bounds}")
            
            if self._required_len != None: 
                if len(v) != self._required_len:
                    raise ValueError(f"Parameter {self.name} cannot have length {len(v)}, must be {self._required_len}")

        elif isinstance(v, (int, float)):
            if not (v >= self._bounds[0] and v <= self._bounds[1]):
                raise ValueError(f"Parameter {self.name} out of bounds {self._bounds}")
            
        elif isinstance(v, str) or (v == None):
            if not v in self._allowed_values and "*" not in self._allowed_values:
                raise ValueError(f"Parameter {self.name} can only have values {self._allowed_values}, not {v}")
            
    def __repr__(self):
        if self.unit is None or self.value is None:
            u = ""
        else:
            u = self.unit
        out = f"<FAST Parameter {self.name}={self.value} {u}>"
        return out

# Master list of all FAST params, including default values, bounds, allowed values, etc.
PARAMS = [
    FastParam("NPXLS", "auto", "Simulation", bounds=[0,None], allowed_values=["auto"]),
    FastParam("DX", "auto", "Simulation", bounds=[0,None], allowed_values=["auto"], unit="m/pixel"),
    FastParam("NITER", 1000, "Simulation", bounds=[1,None]), 
    FastParam("NCHUNKS", 10, "Simulation", bounds=[1,None]),
    FastParam("SUBHARM", False, "Simulation"),
    FastParam("FFTW", False, "Simulation"),
    FastParam("FFTW_THREADS", 1, "Simulation", bounds=[1,None]), 
    FastParam("TEMPORAL", False, "Simulation"), 
    FastParam("DT", 0.001, "Simulation", bounds=[0,None], unit="s"), 
    FastParam("LOGFILE", None, "Simulation", allowed_values=[None, "*"]),
    FastParam("LOGLEVEL", "INFO", "Simulation", allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    FastParam("SEED", None, "Simulation", allowed_values=[None]),
    
    FastParam("W0", "opt", "Ground", bounds=[0,None], allowed_values=["opt"], unit="m"),
    FastParam("D_GROUND", 1.0, "Ground", bounds=[0,None], unit="m"),
    FastParam("OBSC_GROUND", 0.0, "Ground", bounds=[0,None], unit="m"),
    FastParam("AXICON", False, "Ground"),
    FastParam("D_SAT", 0.1, "Satellite", bounds=[0,None], unit="m"),
    FastParam("OBSC_SAT", 0, "Satellite", bounds=[0,None], unit="m"),
    FastParam("H_SAT", 36e6, "Satellite", bounds=[0,None], allowed_values=[None], unit="m"),
    FastParam("L_SAT", None, "Satellite", bounds=[0,None], allowed_values=[None], unit="m"), 
    FastParam("POWER", 1, "Tx/Rx", bounds=[0,None], unit="Watts"),
    FastParam("SMF", True, "Tx/Rx"),

    FastParam("WVL", 1550e-9, "Link", bounds=[0,None], unit="m"),
    FastParam("PROP_DIR", "up", "Link", allowed_values=["up", "down"]),
    FastParam("DTHETA", [4,0], "Link", required_len=2, unit="arcseconds"),
    FastParam("TRANSMISSION", 1, "Link", bounds=[0,1]),

    FastParam("H_TURB", numpy.array([0,10e3]), "Turbulence", bounds=[0,None], unit="m"),
    FastParam("CN2_TURB", numpy.array([100e-15, 100e-15]), "Turbulence", bounds=[0,None], unit="m^1/3"),
    FastParam("WIND_SPD", numpy.array([10,10]), "Turbulence", bounds=[0,None], unit="m/s"),
    FastParam("WIND_DIR", numpy.array([90,0]), "Turbulence", unit="degrees"),
    FastParam("L0", numpy.inf, "Turbulence", bounds=[0,None], unit="m"),
    FastParam("l0", 1e-6, "Turbulence", bounds=[0,None], unit="m"),
    FastParam("ZENITH_ANGLE", 0, "Turbulence", bounds=[0,90], unit="degrees"),

    FastParam("AO_MODE", "AO", "AO", allowed_values=["NOAO", "AO", "TT", "LGSAO"]),
    FastParam("DSUBAP", 0.1, "AO", bounds=[0,None], unit="m"),
    FastParam("TLOOP", 0.001, "AO", bounds=[0,None], unit="s"), 
    FastParam("TEXP", 0.001, "AO", bounds=[0,None], unit="s"),
    FastParam("ALIAS", True, "AO"),
    FastParam("NOISE", 0.0, "AO", bounds=[0,None], unit="?"), 
    FastParam("MODAL", False, "AO"),
    FastParam("MODAL_MULT", 1, "AO", bounds=[0,None]),
    FastParam("ZMAX", None, "AO", bounds=[1,None], allowed_values=[None]),

    FastParam("COHERENT", False, "Comms"),
    FastParam("MODULATION", None, "Comms", allowed_values=[None, "OOK", "BPSK", "QAM", "QPSK", "*"]),
    FastParam("EsN0", None, "Comms", allowed_values=[None], unit="dB")
]

# Default values for all parameters. Each entry contains 
# "PARAMETER": [<default value>, <(allowed types)>, <category>]
# DEFAULTS = {'NPXLS': ['auto', (str, int), "sim"],
#             'DX': ['auto', (str, float), "sim"],
#             'NITER': [1000, (int), "sim"],
#             'SUBHARM': [False, (bool), "sim"],
#             'FFTW': [False, (bool), "sim"],
#             'FFTW_THREADS': [1, (int), "sim"],
#             'NCHUNKS': [10, (int), "sim"],
#             'TEMPORAL': [False, (bool), "sim"],
#             'DT': [0.001, (float), "sim"],
#             'LOGFILE': [None, (None, str), "sim"],
#             'LOGLEVEL': ["INFO", (str), "sim"],
#             'SEED': [None, (None, int), "sim"],   

#             'W0': ["opt", (str, float, int), "rx/tx"],               
#             'D_GROUND': [1.0, (float, int), "rx/tx"],   
#             'OBSC_GROUND': [0, (float, int), "rx/tx"],   
#             'D_SAT': [0.1, (float, int), "rx/tx"],                           
#             'OBSC_SAT': [0, (float, int), "rx/tx"],                     
#             'WVL': [1550e-9, (float), "rx/tx"],        
#             'AXICON': [False, (bool), "rx/tx"],                           
#             'POWER': [1, (float, int), "rx/tx"],                                 
#             'SMF': [True, (bool), "rx/tx"],                                

#             'H_SAT': [36e6, (None, float, int), "turb/link"],
#             'L_SAT': [None, (None, float, int), "turb/link"],
#             'H_TURB': [numpy.array([ 0, 10e3]), (list, numpy.ndarray), "turb/link"],
#             'CN2_TURB': [numpy.array([100e-15, 100e-15]), (list, numpy.ndarray), "turb/link"],
#             'WIND_SPD': [numpy.array([ 10, 10]), (list, numpy.ndarray), "turb/link"],
#             'WIND_DIR': [numpy.array([90., 0.]), (list, numpy.ndarray), "turb/link"],
#             'L0': [numpy.inf, (float, int), "turb/link"],
#             'l0': [1e-06, (float, int), "turb/link"],
#             'ZENITH_ANGLE': [0, (float, int), "turb/link"],
#             'PROP_DIR': ['up', (str), "turb/link"],
#             'DTHETA': [[4,0], (list, numpy.ndarray), "turb/link"],    
#             'TRANSMISSION': [1, (float, int), "turb/link"],

#             'AO_MODE': 'AO',
#             'DSUBAP': 0.02,
#             'TLOOP': 0.001,
#             'TEXP': 0.001,
#             'ALIAS': True,
#             'NOISE': 0.0,
#             'MODAL': False,
#             'MODAL_MULT': 1,
#             'ZMAX': None,

#             'COHERENT': False,
#             'MODULATION': None, 
#             'EsN0': None}

