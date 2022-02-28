import numpy
from . import funcs
from . import ao_power_spectra
from aotools import circle, cn2_to_r0, isoplanaticAngle, coherenceTime
from astropy.io import fits
from tqdm import tqdm

try:
    import pyfftw
    _pyfftw = True
except ImportError:
    _pyfftw = False

class Fast():
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

        if self.fftw and not _pyfftw:
            print('WARNING: fftw flag set but no pyfftw found, defaulting to no fftw')
            self.fftw = False

        self.temporal = params['TEMPORAL']
        self.dt = params['DT']

        if self.Niter % self.Nchunks != 0:
            raise Exception('NCHUNKS must divide NITER without remainder')
        else:
            self.Niter_per_chunk = self.Niter // self.Nchunks

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

        self.freq = SpatialFrequencies(self.Npxls, self.dx)

        self.subharmonics = params['SUBHARM']

        if self.subharmonics:
            self.freq.make_subharm_freqs()

        if self.temporal:
            self.freq.make_temporal_freqs(len(self.h), self.Npxls, self.Niter_per_chunk,
                self.wind_speed, self.wind_dir, self.dt)

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
        self.Gtilt = params['GTILT']

        if self.ao_mode == 'TT_PA':
            # force modal correction with tip/tilt
            self.Zmax = 3
            self.modal = True
            self.modal_mult = 1

        self.lf_mask = ao_power_spectra.mask_lf(self.freq.main, self.Dsubap, 
                    modal=self.modal, modal_mult=self.modal_mult, Zmax=self.Zmax, 
                    D=self.Tx, Gtilt=self.Gtilt)
        self.hf_mask = 1 - self.lf_mask

        if self.subharmonics:
            self.lf_mask_subharm = ao_power_spectra.mask_lf(self.freq.subharm,
                    self.Dsubap, modal=self.modal, modal_mult=self.modal_mult, Zmax=self.Zmax,
                    D=self.Tx, Gtilt=self.Gtilt)

        if self.temporal:
            self.lf_mask_temporal = ao_power_spectra.mask_lf(self.freq.temporal,
                    self.Dsubap, modal=self.modal, modal_mult=self.modal_mult, Zmax=self.Zmax,
                    D=self.Tx, Gtilt=self.Gtilt)

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
            ptype = 'circ'
            # Circular (fully illuminated) aperture
            self.pupil = funcs.compute_pupil(self.Npxls, self.dx, params['Tx'], 
                Tx_obsc=params['Tx_obsc'], ptype=ptype)

        self.pupil_filter = funcs.pupil_filter(self.freq.main, self.pupil, spline=False)

        if self.temporal:
            # compute high-res pupil filter spline for later integration
            fx_max = self.freq.temporal.fx_axis.max()
            fy_max = self.freq.temporal.fy_axis.max()
            f_max = max(fx_max, fy_max)
            dx_req = numpy.pi / f_max
            N_req = int(2*numpy.ceil(2*numpy.pi/(self.freq.main.df * dx_req)/2)) # ensure even
    
            pupil_temporal = funcs.compute_pupil(N_req, dx_req, params['Tx'],
                params['W0'], Tx_obsc=params['Tx_obsc'], ptype=ptype)
            self.freq.make_logamp_freqs(N=N_req, dx=dx_req)
            self.pupil_filter_temporal = funcs.pupil_filter(self.freq.logamp, pupil_temporal, spline=True)

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
            self.freq.main, self.cn2, self.L0, self.l0, C=self.params['C'])

        self.G_ao = ao_power_spectra.G_AO_Jol(
            self.freq.main, self.lf_mask, self.ao_mode, self.h, 
            self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
            self.tloop, self.texp)

        self.aniso_servo_error = funcs.integrate_powerspectrum(funcs.integrate_path(
            self.G_ao * self.turb_powerspec, self.h, layer=self.params['LAYER']) * self.lf_mask * 2 * numpy.pi * self.k**2, self.freq.main.f)

        if self.alias and self.ao_mode is not 'NOAO':
            self.alias_powerspec = ao_power_spectra.Jol_alias_openloop(
                self.freq.main, self.Dsubap, self.cn2, self.lf_mask, self.wind_vector,
                self.texp, self.wvl, 10, 10, self.L0, self.l0)

            self.alias_error = funcs.integrate_powerspectrum(funcs.integrate_path(
                self.alias_powerspec * 2 * numpy.pi * self.k**2, self.h, layer=self.params['LAYER']), self.freq.main.f)
        else:
            self.alias_powerspec = 0.
            self.alias_error = 0.

        if self.noise > 0 and self.ao_mode is not 'NOAO':
            self.noise_powerspec = ao_power_spectra.Jol_noise_openloop(
                self.freq.main, self.Dsubap, self.noise, self.lf_mask)
            self.noise_error = funcs.integrate_powerspectrum(self.noise_powerspec, self.freq.main.f)
        else:
            self.noise_powerspec = 0.
            self.noise_error = 0.

        self.powerspec_per_layer = 2 * numpy.pi * self.k**2 \
            * (self.turb_powerspec * self.G_ao + self.alias_powerspec) + self.noise_powerspec / len(self.h)

        self.powerspec = funcs.integrate_path(self.powerspec_per_layer, h=self.h, layer=self.params['LAYER'])

        self.fitting_error = funcs.integrate_powerspectrum(self.powerspec * self.hf_mask, self.freq.main.f)
        self.phs_var = funcs.integrate_powerspectrum(self.powerspec, self.freq.main.f)
        self.phs_var_weights = funcs.integrate_powerspectrum(self.powerspec_per_layer, self.freq.main.f) / self.phs_var

        # Log-amplitude powerspectrum
        self.logamp_powerspec = ao_power_spectra.logamp_powerspec(self.freq.main, 
            self.h, self.cn2, self.wvl, pupilfilter=self.pupil_filter, layer=self.params['LAYER'], L0=self.L0, l0=self.l0)
        self.logamp_var = funcs.integrate_powerspectrum(self.logamp_powerspec, self.freq.main.f)

        if self.subharmonics:
            self.turb_lo = funcs.turb_powerspectrum_vonKarman(
                self.freq.subharm, self.cn2, self.L0, self.l0, C=self.params['C'])

            self.G_ao_lo = ao_power_spectra.G_AO_Jol(
                self.freq.subharm, self.lf_mask_subharm, 
                self.ao_mode, self.h, self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
                self.tloop, self.texp, self.Dsubap, self.modal, self.modal_mult)

            if self.alias and self.ao_mode is not 'NOAO':
                self.alias_subharm = ao_power_spectra.Jol_alias_openloop(
                    self.freq.subharm, self.Dsubap, 
                    self.cn2, self.lf_mask_subharm, self.wind_vector, self.texp, self.wvl, 10, 10,
                    self.L0, self.l0)
            else:
                self.alias_subharm = 0.

            if self.noise > 0 and self.ao_mode is not 'NOAO':
                self.noise_subharm = ao_power_spectra.Jol_noise_openloop(
                    self.freq.subharm,
                    self.Dsubap, self.noise, self.lf_mask_subharm)
            else:
                self.noise_subharm = 0.

            self.powerspec_subharm_per_layer = 2 * numpy.pi * self.k**2 * \
                (self.turb_lo * self.G_ao_lo + self.alias_subharm) \
                + self.noise_subharm / len(self.h)

            self.powerspec_subharm = funcs.integrate_path(self.powerspec_subharm_per_layer, h=self.h, layer=self.params['LAYER']) 
    
            self.phs_var_subharm = self.powerspec_subharm_per_layer.sum((-1,-2)) * self.freq.subharm.df**2
            self.phs_var_weights_sh = self.phs_var_subharm / self.phs_var_subharm.sum()

        else:
            self.powerspec_subharm = None
            self.phs_var_subharm = None
            self.phs_var_weights_sh = None

        self.temporal_powerspec = None
        self.temporal_logamp_powerspec = None
        self.shifts = None
        self.shifts_sh = None

        if self.temporal:

            f_dot_dx = (self.freq.fx[numpy.newaxis,...] * self.wind_vector[:,0,numpy.newaxis, numpy.newaxis] 
                    + self.freq.fy[numpy.newaxis,...] * self.wind_vector[:,1,numpy.newaxis, numpy.newaxis])

            dts = numpy.arange(0, self.Niter_per_chunk) * self.dt
            self.shifts = numpy.exp(1j * f_dot_dx* dts[..., numpy.newaxis, numpy.newaxis, numpy.newaxis])

            if self.subharmonics:
                f_dot_dx_sh = (self.freq.subharm.fx[numpy.newaxis,...] * self.wind_vector[:,0,numpy.newaxis,numpy.newaxis,numpy.newaxis] 
                            + self.freq.subharm.fy[numpy.newaxis,...] * self.wind_vector[:,1,numpy.newaxis,numpy.newaxis,numpy.newaxis])
                self.shifts_sh = numpy.exp(1j * f_dot_dx_sh * dts[..., numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis])

            self.turb_temporal = funcs.turb_powerspectrum_vonKarman(
                self.freq.temporal, self.cn2, self.L0, self.l0, C=self.params['C'])

            self.G_ao_temporal = ao_power_spectra.G_AO_Jol(
                self.freq.temporal, self.lf_mask_temporal, 
                self.ao_mode, self.h, self.wind_vector, self.dtheta, self.Tx, self.wvl, self.Zmax, 
                self.tloop, self.texp, self.Dsubap, self.modal, self.modal_mult)

            if self.alias and self.ao_mode is not 'NOAO':
                self.alias_temporal = ao_power_spectra.Jol_alias_openloop(
                    self.freq.temporal, self.Dsubap, 
                    self.cn2, self.lf_mask_temporal, self.wind_vector, self.texp, self.wvl, 10, 10,
                    self.L0, self.l0)
            else:
                self.alias_temporal = 0.

            if self.noise > 0 and self.ao_mode is not 'NOAO':
                noise_temporal = ao_power_spectra.Jol_noise_openloop(
                    self.freq.temporal, self.Dsubap, self.noise, self.lf_mask_temporal)
                self.noise_temporal = funcs.integrate_path(noise_temporal, h=self.h, layer=self.params['LAYER'])
                
            else:
                self.noise_temporal = 0.

            temporal_powerspec_beforeintegration = 2 * numpy.pi * self.k**2 * \
                funcs.integrate_path(self.turb_temporal * self.G_ao_temporal + self.alias_temporal, h=self.h, layer=self.params['LAYER']) \
                + self.noise_temporal

            # integrate along y axis
            self.temporal_powerspec = temporal_powerspec_beforeintegration.sum(-2) * self.freq.main.dfy
            # self.temporal_powerspec[len(self.temporal_powerspec)//2] = 0. # ensure the middle is 0!

            temporal_logamp_powerspec_beforeintegration = ao_power_spectra.logamp_powerspec(
                self.freq.temporal, self.h, self.cn2, self.wvl, pupilfilter=self.pupil_filter_temporal, 
                layer=self.params['LAYER'], L0=self.L0, l0=self.l0)

            self.temporal_logamp_powerspec = temporal_logamp_powerspec_beforeintegration.sum(-2) * self.freq.main.dfy

    def compute_scrns(self):

        self.phs = numpy.zeros((self.Niter, *self.powerspec.shape))
        self.logamp = numpy.zeros(self.Niter)

        for i in tqdm(range(self.Nchunks)):
            self.phs[i*self.Niter_per_chunk:(i+1)*self.Niter_per_chunk] = funcs.make_phase_fft(
                self.Niter_per_chunk, self.freq, self.powerspec, self.subharmonics, self.powerspec_subharm, 
                self.dx, self.fftw, self.temporal, self.temporal_powerspec, self.shifts, self.shifts_sh, 
                self.phs_var_weights, self.phs_var_weights_sh)

            self.logamp[i*self.Niter_per_chunk:(i+1)*self.Niter_per_chunk] = \
                funcs.generate_random_coefficients(self.Niter_per_chunk, self.logamp_var, self.temporal, self.temporal_logamp_powerspec).real

        return self.phs

    def compute_I(self, pupil=None):
        if pupil is None:
            pupil = self.pupil * self.fibre_efield

        phase_component = (pupil * numpy.exp(1j * self.phs)).sum((1,2)) * self.dx**2

        self.diffraction_limit = numpy.abs(pupil.sum() * self.dx**2)**2

        self.I = numpy.exp(2 * self.logamp) * numpy.abs(phase_component)**2

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

class SpatialFrequencies():

    def __init__(self, N, dx):

        self.N = N
        self.dx = dx

        self.make_main_freqs(self.N, self.dx)

        # make main spatial frequencies attributes of Frequencies object too
        self.fx = self.main.fx
        self.fy = self.main.fy
        self.fabs = self.main.fabs
        self.f = self.main.f
        self.df = self.main.df

    def make_main_freqs(self, N, dx):
        df = 2*numpy.pi/(N*dx)
        fx_axis = numpy.arange(-N/2., N/2.) * df
        self.main = SpatialFrequencyStruct(fx_axis)

    def make_subharm_freqs(self, pmax=3):

        fx_axes = []
        D = self.dx * self.N
        for i,p in enumerate(range(1,pmax+1)):
            df_lo = 2*numpy.pi/(3**p * D)
            fx_lo = numpy.arange(-1,2) * df_lo
            fx_axes.append(fx_lo)

        self.subharm = SpatialFrequencyStruct(numpy.array(fx_axes))

    def make_temporal_freqs(self, nlayer, Ny, Nx, wind_speed, wind_dir, dt):

        fx_axes = []
        fy_axes = numpy.tile(self.main.f, (nlayer,1))
        
        for i in range(nlayer):
            dx = wind_speed[i] * dt
            # df_temporal = 2 * numpy.pi / (Nx * dx)
            df_temporal = 1 / (Nx * dx) # NOTE Linear spatial frequency here!!!

            # define x axis according to temporal requirements, and y axis 
            # same as the main y axis, since we will integrate over this one
            fx_axis = numpy.arange(-Nx/2, Nx/2) * df_temporal
            fx_axes.append(fx_axis)

        self.temporal = SpatialFrequencyStruct(numpy.array(fx_axes), fy_axes, rot=numpy.radians(wind_dir), freq_per_layer=True)

    def make_logamp_freqs(self, N=None, dx=None):
        if N is None and dx is None:
            self.logamp = self.main
           
        else:
            df = 2*numpy.pi/(N*dx)
            fx_axis = numpy.arange(-N/2., N/2.) * df
            self.logamp = SpatialFrequencyStruct(fx_axis)

class SpatialFrequencyStruct():

    def __init__(self, fx_axis, fy_axis=None, rot=None, freq_per_layer=False):
        
        self.fx_axis = fx_axis
        self.freq_per_layer = freq_per_layer
        if fy_axis is None:
            # x and y axes the same
            self.fy_axis = fx_axis
            self.f = fx_axis
            self.df = fx_axis[...,1]-fx_axis[...,0]
            self.dfx = self.df
            self.dfy = self.df
        else:
            # x and y axes different
            self.fy_axis = fy_axis
            self.dfx = fx_axis[...,1]-fx_axis[...,0]
            self.dfy = fy_axis[...,1]-fy_axis[...,0]

        if self.fx_axis.ndim == 2:
            self._n = self.fx_axis.shape[0]
            self.fx = numpy.zeros((self._n, self.fy_axis.shape[1], self.fx_axis.shape[1]))
            self.fy = numpy.zeros((self._n, self.fy_axis.shape[1], self.fx_axis.shape[1]))

            for i in range(self._n):
                self.fx[i], self.fy[i] = numpy.meshgrid(self.fx_axis[i], self.fy_axis[i])
                if rot is not None:
                    fx_rot = self.fx[i] * numpy.cos(rot[i]) - self.fy[i] * numpy.sin(rot[i])
                    fy_rot = self.fx[i] * numpy.sin(rot[i]) + self.fy[i] * numpy.cos(rot[i])
                    self.fx[i] = fx_rot
                    self.fy[i] = fy_rot

        elif self.fx_axis.ndim == 1:
            self._n = 1
            self.fx, self.fy = numpy.meshgrid(self.fx_axis, self.fy_axis)

            if rot is not None:
                fx_rot = self.fx * numpy.cos(rot) - self.fy * numpy.sin(rot)
                fy_rot = self.fx * numpy.sin(rot) + self.fy * numpy.cos(rot)
                self.fx = fx_rot
                self.fy = fy_rot
        else:
            raise Exception('fx_axis ndim sould be either 1 or 2')

        self.fabs = numpy.sqrt(self.fx**2 + self.fy**2)

    def realspace_sampling(self):
        Nx = self.fx.shape[-1]
        Ny = self.fx.shape[-2]
        dx = 2 * numpy.pi / (Nx * self.dfx)
        dy = 2 * numpy.pi / (Ny * self.dfy)
        return dx, dy