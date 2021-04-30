import numpy
from . import funcs
from . import ao_power_spectra
from aotools import circle
from astropy.io import fits

class FFS():
    def __init__(self, params):
        '''
        Initialise the simulation with a set of parameters.

        TODO: Check that all required params are given at initialisation

        Parameters:
            params (dict): Simulation parameters
        '''
        self.params = params
        self.Niter = params['NITER']

        self.init_atmos(self.params)
        self.init_frequency_grid(self.params)
        self.init_beam_params(self.params)
        self.init_ao_params(self.params)
        self.init_pupil_mask(self.params)

        self.compute_powerspec()

    def run(self):
        self.compute_scrns()
        I = self.compute_I()
        return I

    def init_frequency_grid(self, params):
        if params['DX'] is 'auto':
            self.dx = params['DSUBAP'] / 2 # Nyquist sample WFS subaperture
        else:
            self.dx = params['DX']
        
        if params['NPXLS'] is 'auto':
            nyq_paa = numpy.pi / (self.h[-1] * self.paa/206265.) # nyquist sampling of highest spatial frequency required ()
            nyq_temp = numpy.pi / (max(self.wind_speed) * params['TLOOP'])
            nyq = numpy.min([nyq_paa, nyq_temp])
            self.Npxls = int(2*numpy.ceil(2*numpy.pi/(nyq * self.dx)/2)) # ensure even
        else:
            self.Npxls = params['NPXLS']

        self.fx, self.fy, self.fabs, self.f = funcs.f_grid_dx(self.Npxls, self.dx)
        self.df = self.f[1] - self.f[0]

        self.subharmonics = params['SUBHARM']
        if self.subharmonics:
            self.fx_subharm = numpy.zeros((3,3,3))
            self.fy_subharm = numpy.zeros((3,3,3))
            self.fabs_subharm = numpy.zeros((3,3,3))
            D = self.dx * self.Npxls
            for i,p in enumerate(range(1,4)):
                df_lo = 2*numpy.pi/(3**p * D)
                fx_lo = numpy.arange(-1,2) * df_lo
                fx_lo, fy_lo = numpy.meshgrid(fx_lo, fx_lo)
                fabs_lo = numpy.sqrt(fx_lo**2 + fy_lo**2)
                self.fx_subharm[i] = fx_lo
                self.fy_subharm[i] = fy_lo
                self.fabs_subharm[i] = fabs_lo
        else:
            self.fx_subharm = self.fy_subharm = self.fabs_subharm = None

    def init_atmos(self, params):
        self.zenith_correction = self.calc_zenith_correction(params['ZENITH_ANGLE'])
        self.h = params['H_TURB'] * self.zenith_correction
        self.cn2 = params['CN2_TURB'] * self.zenith_correction
        self.L = params['H_SAT'] * self.zenith_correction
        self.wind_speed = params['WIND_SPD']
        self.wind_dir = params['WIND_DIR']
        self.dtheta = params['DTHETA']
        self.paa = numpy.sqrt(self.dtheta[0]**2 + self.dtheta[1]**2)
        self.wind_vector = (self.wind_speed * 
            numpy.array([numpy.cos(numpy.radians(self.wind_dir)),
                         numpy.sin(numpy.radians(self.wind_dir))])).T

    def init_beam_params(self, params):
        self.W0 = params['W0']
        self.F0 = params['F0']
        self.wvl = params['WVL']
        self.k = 2*numpy.pi/self.wvl

        self.Theta_0, self.Lambda_0, self.Theta, self.Lambda, self.Theta_bar = \
            funcs.calc_gaussian_beam_parameters(self.L, params['F0'], params['W0'], params['WVL'])
        self.W = params['W0'] * numpy.sqrt(self.Theta_0**2 + self.Lambda_0**2)

        self.Tx = params['Tx']
        self.Tx_obsc = params['Tx_obsc']
        self.Rx = params['Rx']

    def init_ao_params(self, params):
        self.ao_mode = params['AO_MODE']
        self.Dsubap = params['DSUBAP']
        self.tloop = params['TLOOP']
        self.texp = params['TEXP']
        self.Zmax = params['ZMAX']

    def init_pupil_mask(self, params):
        if params['PROP_DIR'] is 'up':
            # Gaussian aperture
            if params['AXICON']:
                ptype = 'axicon'
            else:
                ptype = 'gauss'

            self.pupil = funcs.compute_pupil(self.Npxls, self.dx, params['Tx'], 
                params['W0'], params['Tx_obsc'], ptype=ptype)
        else:
            # Circular (fully illuminated) aperture
            self.pupil = funcs.compute_pupil(self.Npxls, self.dx, params['Tx'], 
                Tx_obsc=params['Tx_obsc'], ptype='circ')
                
        return self.pupil

    def compute_powerspec(self):
        self.turb_powerspec = funcs.turb_powerspectrum_vonKarman(
            self.fabs, self.cn2, self.params['L0'], self.params['l0'], C=self.params['C'])

        self.G_ao = ao_power_spectra.G_AO_Jol(
            self.fabs, self.fx, self.fy, self.ao_mode, self.h, 
            self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
            self.tloop, self.texp, self.Dsubap)

        self.powerspec = 2 * numpy.pi * self.k**2 * funcs.integrate_path(
            self.turb_powerspec * self.G_ao, self.h, layer=self.params['LAYER'])

        if self.subharmonics:
            turb_lo = funcs.turb_powerspectrum_vonKarman(
                self.fabs_subharm, self.cn2, self.params['L0'], self.params['l0'], C=self.params['C'])

            G_ao_lo = ao_power_spectra.G_AO_Jol(
                self.fabs_subharm, self.fx_subharm, self.fy_subharm, self.ao_mode, self.h, 
                self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
                self.tloop, self.texp, self.Dsubap)

            self.powerspec_subharm = 2 * numpy.pi * self.k**2 * funcs.integrate_path(
                turb_lo * G_ao_lo, self.h, layer=self.params['LAYER'])
        else:
            self.powerspec_subharm = None

    def compute_scrns(self):
        self.phs = funcs.make_phase_fft(
            self.Niter, self.powerspec, self.df, self.subharmonics,
            self.powerspec_subharm, self.fx_subharm, self.fy_subharm, 
            self.fabs_subharm, self.dx)

        return self.phs

    def compute_I(self, pupil=None):
        if pupil is None:
            pupil = self.pupil

        logamp_var = funcs.logamp_var(pupil, self.dx, self.h, self.cn2, self.wvl,
            self.params['L0'], self.params['l0'])
        self.rand_logamp = numpy.random.normal(
            loc=0, scale=numpy.sqrt(logamp_var), size=(self.Niter,))

        phase_component = (pupil * numpy.exp(1j * self.phs)).sum((1,2)) * self.dx**2

        self.I = numpy.exp(2 * self.rand_logamp) * numpy.abs(phase_component)**2

        self.I /= (self.wvl * self.L)**2

        return self.I

    def calc_zenith_correction(self, zenith_angle):
        zenith_angle_rads = numpy.radians(zenith_angle)
        gamma = 1/numpy.cos(zenith_angle_rads)
        return gamma

    def make_header(self, params):
        hdr = fits.Header()
        hdr['ZENITH'] = params['ZENITH_ANGLE']
        hdr['WVL'] = int(params['WVL']*1e9)
        if numpy.isinf(params['L0']):
            hdr['OTRSCALE'] = str(params['L0'])
        else:
            hdr['OTRSCALE'] = params['L0']
        hdr['INRSCALE'] = params['l0']
        hdr['POWER'] = params['POWER'] 
        hdr['PAA'] = self.paa
        hdr['TLOOP'] = params['TLOOP'] 
        hdr['TEXP'] = params['TEXP']
        hdr['DSUBAP'] = params['DSUBAP'] 
        hdr['TX'] = params['Tx']
        hdr['TX_OBSC'] = params['Tx_obsc']
        hdr['AXICON'] = str(params['AXICON'])
        hdr['H_SAT'] = params['H_SAT']
        hdr['DX'] = self.dx
        hdr['NPXLS'] = self.Npxls
        hdr['NITER'] = self.Niter
        return hdr

    def save(self, fname, **kwargs):
        hdr = self.make_header(self.params) 
        fits.writeto(fname, self.I, header=hdr, **kwargs)
