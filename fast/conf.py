'''
Needs to read in python/YAML file and spit out a dictionary, taking into account 
default values for certain parameters.
'''
import importlib.util
import numpy


class ConfigParser():

    def __init__(self, fname):

        self.fname = fname
        self.config = {}
        self.defaults = {}

        self.set_defaults()
        self.load(fname)

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
                print(f"Config parameter {key} not defined in {self.fname}, setting default value of {self.defaults[key]}")
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
            'W0': 0.1,
            'F0': numpy.inf,
            'Tx': 1.0,
            'Tx_obsc': 0.35,
            'Rx': 0.01,
            'DTHETA': [0, 0],
            'WVL': 1550e-9,
            'AXICON': False,
            'POWER': 1,
            'SMF': True,
            'COHERENT': False,
            'L_SAT': 36e6,
            'H_TURB': numpy.array([ 0, 10e3]),
            'CN2_TURB': numpy.array([100e-15, 100e-15]),
            'WIND_SPD': numpy.array([ 10, 10]),
            'WIND_DIR': numpy.array([90., 0.]),
            'L0': numpy.inf,
            'l0': 1e-06,
            'C': 2*numpy.pi,
            'ZENITH_ANGLE': 0,
            'PROP_DIR': 'down',
            'AO_MODE': 'AO',
            'DSUBAP': 0.02,
            'TLOOP': 0.001,
            'TEXP': 0.001,
            'ALIAS': True,
            'NOISE': 0.0,
            'MODAL': False,
            'MODAL_MULT': 1,
            'ZMAX': None,
            'GTILT': False,
            'MODULATION': None, 
            'N0': 0}

