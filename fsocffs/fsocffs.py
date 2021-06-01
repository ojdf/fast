import numpy
from . import funcs
from . import ao_power_spectra
from aotools import circle, cn2_to_r0, isoplanaticAngle, coherenceTime
from astropy.io import fits
from tqdm import tqdm

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
        self.Nchunks = params['NCHUNKS']
        self.fftw = params['FFTW']

        if self.Niter % self.Nchunks != 0:
            raise Exception('NCHUNKS must divite NITER without remainder')

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
            nyq_Npxls = int(2*numpy.ceil(2*numpy.pi/(nyq * self.dx)/2)) # ensure even
            ap_Npxls = int(2*numpy.ceil(params['Tx']/self.dx/2))
            self.Npxls = numpy.max([nyq_Npxls, ap_Npxls])
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

        self.r0 = cn2_to_r0(params['CN2_TURB'].sum(), lamda=500e-9)
        self.theta0 = isoplanaticAngle(params['CN2_TURB'], params['H_TURB'], lamda=500e-9)
        self.tau0 = coherenceTime(params['CN2_TURB'], params['WIND_SPD'], lamda=500e-9)
        self.L0 = params['L0']
        self.l0 = params['l0']

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
        self.alias = params['ALIAS']
        self.noise = params['NOISE']
        self.modal = params['MODAL']
        self.modal_mult = params['MODAL_MULT']

        if self.ao_mode == 'TT_PA':
            # force modal correction with tip/tilt
            self.Zmax = 3
            self.modal = True
            self.modal_mult = 1

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

        if params['SMF']:
            # compute optimal SMF parameters
            self.smf = True
            self.fibre_efield = funcs.optimize_fibre(self.pupil, self.dx)
        else:
            self.smf = False
            self.fibre_efield = 1.

        return self.pupil

    def compute_powerspec(self):
        self.turb_powerspec = funcs.turb_powerspectrum_vonKarman(
            self.fabs, self.cn2, self.L0, self.l0, C=self.params['C'])

        self.G_ao = ao_power_spectra.G_AO_Jol(
            self.fabs, self.fx, self.fy, self.ao_mode, self.h, 
            self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
            self.tloop, self.texp, self.Dsubap, self.modal, self.modal_mult)

        if self.alias:
            self.alias_powerspec = ao_power_spectra.Jol_alias_openloop(
                self.fabs, self.fx, self.fy, self.Dsubap, self.cn2, self.wind_vector,
                self.texp, self.wvl, 10, 10, self.L0, self.l0, self.modal, self.modal_mult,
                self.Zmax, self.Tx)
        else:
            self.alias_powerspec = 0.

        if self.noise > 0:
            self.noise_powerspec = ao_power_spectra.Jol_noise_openloop(
                self.fabs, self.fx, self.fy, self.Dsubap, self.noise, self.modal, self.modal_mult, 
                self.Zmax, self.Tx)
        else:
            self.noise_powerspec = 0.

        self.powerspec = 2 * numpy.pi * self.k**2 * funcs.integrate_path(
            self.turb_powerspec * self.G_ao + self.alias_powerspec, self.h, 
            layer=self.params['LAYER']) + self.noise_powerspec

        if self.subharmonics:
            self.turb_lo = funcs.turb_powerspectrum_vonKarman(
                self.fabs_subharm, self.cn2, self.L0, self.l0, C=self.params['C'])

            self.G_ao_lo = ao_power_spectra.G_AO_Jol(
                self.fabs_subharm, self.fx_subharm, self.fy_subharm, self.ao_mode, self.h, 
                self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
                self.tloop, self.texp, self.Dsubap, self.modal, self.modal_mult)

            if self.alias:
                self.alias_subharm = ao_power_spectra.Jol_alias_openloop(
                    self.fabs_subharm, self.fx_subharm, self.fy_subharm, self.Dsubap, 
                    self.cn2, self.wind_vector, self.texp, self.wvl, 10, 10,
                    self.L0, self.l0, self.modal, self.modal_mult, self.Zmax, self.Tx)
            else:
                self.alias_subharm = 0.

            if self.noise > 0:
                self.noise_subharm = ao_power_spectra.Jol_noise_openloop(
                    self.fabs_subharm, self.fx_subharm, self.fy_subharm, 
                    self.Dsubap, self.noise, self.modal, self.modal_mult, 
                    self.Zmax, self.Tx)
            else:
                self.noise_subharm = 0.

            self.powerspec_subharm = 2 * numpy.pi * self.k**2 * funcs.integrate_path(
                self.turb_lo * self.G_ao_lo + self.alias_subharm, self.h, layer=self.params['LAYER']) \
                + self.noise_subharm
        else:
            self.powerspec_subharm = None

    def compute_scrns(self):
        self.phs = numpy.zeros((self.Niter, *self.powerspec.shape))

        chunk_iters = int(self.Niter/self.Nchunks)

        for i in tqdm(range(self.Nchunks)):
            self.phs[i*chunk_iters:(i+1)*chunk_iters] = funcs.make_phase_fft(
                chunk_iters, self.powerspec, self.df, self.subharmonics,
                self.powerspec_subharm, self.fx_subharm, self.fy_subharm, 
                self.fabs_subharm, self.dx, self.fftw)

        return self.phs

    def compute_I(self, pupil=None):
        if pupil is None:
            pupil = self.pupil * self.fibre_efield

        logamp_var = funcs.logamp_var(pupil, self.dx, self.h, self.cn2, self.wvl,
            self.L0, self.l0)
        self.rand_logamp = numpy.random.normal(
            loc=0, scale=numpy.sqrt(logamp_var), size=(self.Niter,))

        phase_component = (pupil * numpy.exp(1j * self.phs)).sum((1,2)) * self.dx**2

        self.diffraction_limit = numpy.abs(pupil.sum() * self.dx**2)**2

        self.I = numpy.exp(2 * self.rand_logamp) * numpy.abs(phase_component)**2

        if self.params['PROP_DIR'] is 'up':
            # Far field intensity
            self.I /= (self.wvl * self.L)**2
            self.diffraction_limit /= (self.wvl * self.L)**2

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
        hdr['ALIAS'] = str(params['ALIAS'])
        hdr['NOISE'] = params['NOISE']
        hdr['TX'] = params['Tx']
        hdr['TX_OBSC'] = params['Tx_obsc']
        hdr['AXICON'] = str(params['AXICON'])
        hdr['H_SAT'] = params['H_SAT']
        hdr['DX'] = self.dx
        hdr['NPXLS'] = self.Npxls
        hdr['NITER'] = self.Niter
        hdr['R0'] = self.r0
        hdr['THETA0'] = self.theta0
        hdr['TAU0'] = self.tau0
        return hdr

    def save(self, fname, **kwargs):
        hdr = self.make_header(self.params) 
        fits.writeto(fname, self.I, header=hdr, **kwargs)
