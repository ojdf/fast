'''
Needs to read in python/YAML file and spit out a dictionary, taking into account 
default values for certain parameters.
'''
import importlib.util
import numpy
import logging

logger = logging.getLogger(__name__)

class ConfigParser():

    def __init__(self, fname_or_dict):
        
        if type(fname_or_dict) == dict:
            self.config = fname_or_dict
            self.fname = None
        elif type(fname_or_dict) == str:
            self.fname = fname_or_dict
            self.config = {}
            self.load(fname_or_dict)
        else:
            raise Exception("Either config file name or params dict required")

        self.defaults = {}
        self.set_defaults()

        self.check()
        
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
            self.config = conf_module.p
            return

        raise Exception("Require .py config file")

    def check(self):
        '''
        Check loaded config dictionary for missing values. If they exist, replace 
        with defualt values
        '''
        for key in self.defaults.keys():
            try:
                self.config[key]
            except KeyError:
                logger.warning(f"Config parameter {key} not defined in {self.fname}, setting default value of {self.defaults[key]}")
                self.config[key] = self.defaults[key]

    def set_defaults(self):
        '''
        Set defualt values for all config parameters from global DEFAULTS dictionary
        '''
        self.defaults = DEFAULTS


DEFAULTS = {'NPXLS': 'auto',
            'DX': 'auto',
            'NITER': 1000,
            'SUBHARM': False,
            'FFTW': False,
            'FFTW_THREADS': 1,
            'NCHUNKS': 10,
            'TEMPORAL': False,
            'DT': 0.001,
            'LOGFILE': None,
            'LOGLEVEL': "INFO",
            'SEED': None,   

            'W0': "opt",               
            'D_GROUND': 1.0,   
            'OBSC_GROUND': 0,   
            'D_SAT': 0.1,                           
            'OBSC_SAT': 0,                     
            'WVL': 1550e-9,        
            'AXICON': False,                           
            'POWER': 1,                                 
            'SMF': True,                                

            'H_SAT': 36e6,
            'L_SAT': None,
            'H_TURB': numpy.array([ 0, 10e3]),
            'CN2_TURB': numpy.array([100e-15, 100e-15]),
            'WIND_SPD': numpy.array([ 10, 10]),
            'WIND_DIR': numpy.array([90., 0.]),
            'L0': numpy.inf,
            'l0': 1e-06,
            'ZENITH_ANGLE': 0,
            'PROP_DIR': 'up',
            'DTHETA': [4,0],    
            'TRANSMISSION': 1,

            'AO_MODE': 'AO',
            'DSUBAP': 0.02,
            'TLOOP': 0.001,
            'TEXP': 0.001,
            'ALIAS': True,
            'NOISE': 0.0,
            'MODAL': False,
            'MODAL_MULT': 1,
            'ZMAX': None,

            'COHERENT': False,
            'MODULATION': None, 
            'EsN0': None}

