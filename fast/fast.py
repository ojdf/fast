import numpy
from . import funcs
from . import ao_power_spectra
from . import conf
from aotools import circle, cn2_to_r0, isoplanaticAngle, coherenceTime, fouriertransform
from astropy.io import fits
from tqdm import tqdm
import logging
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)

try:
    import pyfftw
    _pyfftw = True
except ImportError:
    _pyfftw = False


class Fast():
    """Base class of the FAST simulation. 

    This object is should be created in order to initialise and run simulations,
    providing a ``params`` object with simulation parameters/configuration. The 
    ``params`` object may either be a string, in which case it should point to 
    the location of a config file, i.e. 
    
        sim = fast.Fast("/path/to/config.py")

    or a dictionary containing the relevant config information, i.e. 
    
        | p = config_dict 
        | sim = fast.Fast(p)
    
    An example params file is located in ``test/test_params.py``. 

    Once created, the simulation will have computed everything required apart 
    from the Monte Carlo phase screens. This is the most computationally expensive 
    operation and is started by calling the ``run`` method on the object, i.e.

        I = sim.run()

    This will return a 1D numpy array of values, which are also stored in ``sim.I``. 
    These values are normalised to the diffraction limit (i.e. no turbulence). To 
    obtain absolute values of e.g. received power, one can multiply by the 
    ``sim.diffraction_limit`` parameter which is precomputed based on the link 
    parameters. A full link budget is stored in ``sim.link_budget``.

    Attributes:
        I (ndarray): 1D data array storing results of simulation, normalised to 
            diffraction limit. Created after running ``sim.run()``.

        link_budget (dict): Dictionary containing link budget terms, calculated 
            at simulation initialisation.

        powerspec (ndarray): 2D AO-corrected residual phase power spectrum. 

        params (dict): Parameter dictionary

        r0 (float): Fried parameter @ 500nm @ zenith

        theta0 (float): Isoplanatic angle @ 500nm @ zenith

        tau0 (float): Coherence time @ 500nm @ zenith

    Args:
        params (str): file name of config file 
        OR
        params (dict): config dict
    """
    def __init__(self, params):

        self.conf = conf.ConfigParser(params)
        self.params = self.conf.config

        self.Niter = self.params['NITER']
        self.Nchunks = self.params['NCHUNKS']
        self.fftw = self.params['FFTW']
        self.nthreads = self.params['FFTW_THREADS']
        self.seed = self.params['SEED']
        if self.seed != None:
            self.set_seed(self.seed)

        self.temporal = self.params['TEMPORAL']
        self.dt = self.params['DT']

        if self.Niter % self.Nchunks != 0:
            raise Exception('NCHUNKS must divide NITER without remainder')
        else:
            self.Niter_per_chunk = self.Niter // self.Nchunks

        if not (self.Niter_per_chunk % 2 == 0) and not self.temporal:
            raise Exception('NITER/NCHUNKS must be even number')

        self.init_logging()
        self.init_atmos()
        self.init_beam_params()
        self.init_frequency_grid()
        self.init_ao_params()
        self.init_pupil_mask()
        self.init_phs_logamp()

        self.compute_link_budget()
        self.compute_powerspec()

        self.fftw_objs = None
        if self.fftw:
            if not _pyfftw:
                logger.warning('fftw flag set but no pyfftw found, defaulting to no fftw')
                self.fftw = False
            else:
                # Create fftw objects for later
                self.init_fftw()

    def run(self):

        if self.params['COHERENT']:
            I = numpy.zeros((self.Nchunks, self.Niter_per_chunk), dtype=complex)
        else:
            I = numpy.zeros((self.Nchunks, self.Niter_per_chunk))

        logger.debug("Compute log amplitude values")
        self.compute_logamp()

        if not self.temporal:
            phsfunc = self.compute_phs
        else:
            phsfunc = self.compute_phs_temporal
        
        for i in tqdm(range(self.Nchunks)):
            logger.debug(f"Compute phase for chunk {i+1}")
            phsfunc(chunk=i)
            logger.debug(f"Compute detector for chunk {i+1}")
            I[i] = self.compute_detector(chunk=i)

        self.result = FastResult(I.flatten(), self.diffraction_limit)
        self.I = self.result.power # backwards compatibility

        logger.info(self.result)
        return self.result

    def init_logging(self):
        logging.basicConfig(filename=self.params['LOGFILE'], 
                            level=logging.getLevelName(self.params['LOGLEVEL']),
                            format="[%(levelname)s] %(name)s.%(funcName)s | %(message)s")

    def init_frequency_grid(self):

        logger.info("Initialising spatial frequencies")

        if self.params['DX'] == 'auto':
            # Nyquist sample either WFS subap or r0, or ensure 10 pixels 
            # across pupil (required for very small launched beams)
            dx_subap = self.params['DSUBAP'] / 2
            dx_r0 = self.r0_los / 2
            dx_pupil = self.D_ground / 10
            self.dx = numpy.min([dx_subap, dx_r0, dx_pupil])

            if self.params['AO_MODE'] == 'NOAO':
                # Set the number of pixels to be based on turbulence only
                self.dx = self.r0_los / 2

            logger.info(f"Auto set DX to {self.dx}")
        else:
            self.dx = self.params['DX']
        
        if self.params['NPXLS'] == 'auto':
            # Nyquist sample highest spatial frequency required for aniso-servo PSD
            nyq_aniso = numpy.pi / (self.h[-1] * self.paa/206265.) 
            nyq_servo = numpy.pi / (max(self.wind_speed) * self.params['TLOOP'])

            # 10 pixels across AO corrected region (arbitrary...)
            nyq_fitting = numpy.pi / self.params['DSUBAP'] / 5

            nyq = numpy.min([nyq_aniso, nyq_servo, nyq_fitting])
            nyq_Npxls = int(2*numpy.ceil(2*numpy.pi/(nyq * self.dx)/2)) # ensure even

            # Make sure enough pixels so aperture is not clipped!
            ap_Npxls = int(2*numpy.ceil(self.params['D_GROUND']/self.dx/2)) + 2 
            
            if self.params['AO_MODE'] == 'NOAO' and not numpy.isinf(self.params['L0']):
                # Choose number of pixels so that the phase screen is 2x outer scale
                L0_Npxls = int(2 * numpy.ceil((self.params['L0'] * 2) / self.dx) / 2)
            else:
                L0_Npxls = 0

            if self.params['TEMPORAL']:
                # need enough pixels to not wrap 
                temporal_Npxls = int(self.params['WIND_SPD'].max() * self.params['DT'] * self.params['NITER'] / self.params['DX'] / 2)
            else:
                temporal_Npxls = 0

            self.Npxls = numpy.max([nyq_Npxls, ap_Npxls, L0_Npxls, temporal_Npxls])

            logger.info(f"Auto set NPXLS to {self.Npxls}")

        else:
            self.Npxls = self.params['NPXLS']

            # Warning if in temporal mode and not enough pixels 
            if self.params['TEMPORAL']:
                temporal_Npxls = int(self.params['WIND_SPD'].max() * self.params['DT'] * self.params['NITER'] / self.params['DX'] / 2)
                if self.Npxls < temporal_Npxls:
                    logger.warning("NPXLS is likely too small -- some periodicity may occur in your resulting time series")
                    logger.warning(f"Current value: {self.Npxls}")
                    logger.warning(f"Recommended value: {temporal_Npxls}")

        if self.Npxls > 2048:
            logger.warning(f"NPXLS is large ({self.Npxls}) and may cause very high memory usage")

        self.Npxls_pup = int(numpy.ceil(self.D_ground/self.dx)) + 2

        self.freq = SpatialFrequencies(self.Npxls, self.dx)

        self.subharmonics = self.params['SUBHARM']

        if self.temporal:
            self.freq.make_temporal_freqs(len(self.h), self.Npxls, self.Niter,
                self.wind_speed, self.wind_dir, self.dt)
            
            # also turn off subharmonics for temporal mode 
            if self.subharmonics:
                logger.info("SUBHARM not used in TEMPORAL mode")
                self.subharmonics = False
        
        if self.subharmonics:
            self.freq.make_subharm_freqs()

    def init_atmos(self):

        logger.info("Initialising atmosphere")

        self.zenith_correction = self.calc_zenith_correction(self.params['ZENITH_ANGLE'])
        self.h = self.params['H_TURB'] * self.zenith_correction
        self.cn2 = self.params['CN2_TURB'] * self.zenith_correction

        # If L_SAT is defined, use that otherwise use H_SAT and zenith correct
        if self.params['L_SAT'] != None:
            self.L = self.params['L_SAT']
        else:
            self.L = funcs.l_path(self.params['H_SAT'], self.params['ZENITH_ANGLE'])

        # PAA 
        self.dtheta = self.params['DTHETA']
        self.paa = numpy.sqrt(self.dtheta[0]**2 + self.dtheta[1]**2)

        # Wind calculations
        self.wind_dir = self.params['WIND_DIR']
        try:
            self.wind_dir = [(x - self.params['AZIMUT_SAT'])%380 for x in self.wind_dir]
        except KeyError:
            pass 
        self.wind_vector = (self.params['WIND_SPD'] * numpy.array([numpy.cos(numpy.radians(self.wind_dir)),
                        (numpy.sin(numpy.radians(self.wind_dir)))/self.zenith_correction])).T
        try:
            self.wind_correction = funcs.calculate_wind_correction(self.h, self.params['ANISO_DL'], self.params['TLOOP'])
            self.wind_vector += self.wind_correction
        except KeyError:
            pass
        self.wind_speed = numpy.sqrt(self.wind_vector[:,0]**2 + self.wind_vector[:,1]**2)


        # Atmospheric parameters at zenith, at 500 nm
        self.r0 = cn2_to_r0(self.params['CN2_TURB'].sum(), lamda=500e-9)
        self.theta0 = isoplanaticAngle(self.params['CN2_TURB'], self.params['H_TURB'], lamda=500e-9)
        self.tau0 = coherenceTime(self.params['CN2_TURB'], self.params['WIND_SPD'], lamda=500e-9)

        # Atmospheric parameters along line of sight (los), at laser wavelength
        self.r0_los = cn2_to_r0(self.cn2.sum(), lamda=self.params['WVL'])
        self.theta0_los = isoplanaticAngle(self.cn2, self.h, lamda=self.params['WVL'])
        self.tau0_los = coherenceTime(self.cn2, self.wind_speed, lamda=self.params['WVL'])

        self.L0 = self.params['L0']
        self.l0 = self.params['l0']

    def init_beam_params(self):

        logger.info("Initialising beam parameters")

        self.power = self.params['POWER']
        self.W0 = self.params['W0']
        self.F0 = numpy.inf # hard-coded, always launch collimated beam
        self.wvl = self.params['WVL']
        self.k = 2*numpy.pi/self.wvl

        # self.Theta_0, self.Lambda_0, self.Theta, self.Lambda, self.Theta_bar = \
        #     funcs.calc_gaussian_beam_parameters(self.L, self.F0, self.W0, self.wvl)
        # self.W = self.W0 * numpy.sqrt(self.Theta_0**2 + self.Lambda_0**2)

        self.D_ground = self.params['D_GROUND']
        self.obsc_ground = self.params['OBSC_GROUND']
        self.D_sat = self.params['D_SAT']
        self.obsc_sat = self.params['OBSC_SAT']

    def init_ao_params(self):

        logger.info("Initialising AO parameters")

        self.ao_mode = self.params['AO_MODE']
        self.Dsubap = self.params['DSUBAP']
        self.tloop = self.params['TLOOP']
        self.texp = self.params['TEXP']
        self.Zmax = self.params['ZMAX']
        self.alias = self.params['ALIAS']
        self.noise = self.params['NOISE']
        self.modal = self.params['MODAL']
        self.modal_mult = self.params['MODAL_MULT']

        if self.ao_mode == 'TT':
            # force modal correction with tip/tilt
            self.Zmax = 3
            self.modal = True
            self.modal_mult = 1

        self.lf_mask = ao_power_spectra.mask_lf(self.freq.main, self.Dsubap, 
                    modal=self.modal, modal_mult=self.modal_mult, Zmax=self.Zmax, 
                    D=self.D_ground)
        self.hf_mask = 1 - self.lf_mask

        if self.subharmonics:
            self.lf_mask_subharm = ao_power_spectra.mask_lf(self.freq.subharm,
                    self.Dsubap, modal=self.modal, modal_mult=self.modal_mult, Zmax=self.Zmax,
                    D=self.D_ground)

        if self.temporal:
            self.lf_mask_temporal = ao_power_spectra.mask_lf(self.freq.temporal,
                    self.Dsubap, modal=self.modal, modal_mult=self.modal_mult, Zmax=self.Zmax,
                    D=self.D_ground)

    def init_pupil_mask(self):

        logger.info("Initialising pupil mask")

        # NOTE setting satellite pupil sampling to be fixed 32 pixels here, should 
        # probably change this 
        self.dx_sat = self.D_sat/32


        # if self.params['PROP_DIR'] == 'up':
        #     # Gaussian aperture at ground
        #     if self.params['AXICON']:
        #         ptype = 'axicon'
        #     else:
        #         ptype = 'gauss'

        #     puptmp = funcs.compute_pupil(self.Npxls, self.dx, self.D_ground, 
        #         self.W0, self.obsc_ground, ptype=ptype)
            
        #     if self.W0 == "opt":
        #         self.pupil, self.W0 = puptmp
        #     else:
        #         self.pupil = puptmp

        #     # Circ aperture at satellite
        #     self.pupil_sat = funcs.compute_pupil(32, self.dx_sat, self.D_sat,
        #         Tx_obsc=self.obsc_sat, ptype='circ') 

        # else:
        #     ptype = 'circ'
        #     # Circular (fully illuminated) aperture at ground
        #     self.pupil = funcs.compute_pupil(self.Npxls, self.dx, self.D_ground, 
        #         Tx_obsc=self.obsc_ground, ptype=ptype)

        #     # Gaussian aperture at satellite (NOTE hard coded 32 pxls)
        #     pupsattmp = funcs.compute_pupil(32, self.dx_sat, self.D_sat, 
        #         W0=self.W0, Tx_obsc=self.obsc_sat, ptype='gauss')
            
        #     if self.W0 == "opt":
        #         self.pupil_sat, self.W0 = pupsattmp
        #     else:
        #         self.pupil_sat = pupsattmp

        ptype = 'gauss'
        if self.params['AXICON']:
            ptype = 'axicon'

        self.pupil = funcs.compute_pupil(self.Npxls, self.dx, self.D_ground, self.obsc_ground)
        self.pupil_sat = funcs.compute_pupil(32, self.dx_sat, self.D_sat, self.obsc_sat)
        
        self.pupil_mode, self.W0 = funcs.compute_gaussian_mode(self.pupil, self.dx, self.W0, D=self.D_ground, 
                                                      obsc=self.obsc_ground, ptype=ptype)
        self.pupil_mode_sat, self.W0_sat = funcs.compute_gaussian_mode(self.pupil_sat, self.dx_sat, "opt", ptype="gauss")


        self.pupil_filter = funcs.pupil_filter(self.freq.main, self.pupil * self.pupil_mode, spline=False)

        # Cut out only the actual pupil 
        self.pup_coords = numpy.array((numpy.arange((self.Npxls-self.Npxls_pup)//2,(self.Npxls+self.Npxls_pup)//2), numpy.arange((self.Npxls-self.Npxls_pup)//2,(self.Npxls+self.Npxls_pup)//2))).astype(int)
        self.pupil = self.pupil[self.pup_coords[0],:][:,self.pup_coords[1]]
        self.pupil_mode = self.pupil_mode[self.pup_coords[0],:][:,self.pup_coords[1]]

        if self.temporal:
            # compute high-res pupil filter spline for later integration
            fx_max = self.freq.temporal.fx_axis.max()
            fy_max = self.freq.temporal.fy_axis.max()
            f_max = max(fx_max, fy_max)
            dx_req = numpy.pi / f_max
            N_req = int(2*numpy.ceil(2*numpy.pi/(self.freq.main.df * dx_req)/2)) # ensure even
    
            pupil_temporal = funcs.compute_pupil(N_req, dx_req, self.D_ground, self.obsc_ground, Ny=2*self.Npxls_pup)
            mode_temporal, _ = funcs.compute_gaussian_mode(pupil_temporal, dx_req, W0=self.W0, ptype="gauss")
            self.freq.make_logamp_freqs(Nx=N_req, dx=dx_req, Ny=2*self.Npxls_pup, dy=self.dx)
            self.pupil_filter_temporal = funcs.pupil_filter(self.freq.logamp, pupil_temporal * mode_temporal, spline=True)

        # self.smf = self.params['SMF']
        # self.fibre_efield = 1.
        # self.fibre_efield_sat = 1.
        # if self.smf:
        #     # compute optimal SMF parameters at Rx
        #     if self.params['PROP_DIR'] == "up":
        #         self.fibre_efield_sat = funcs.optimize_fibre(self.pupil_sat, self.dx_sat)
        #     else:
        #         self.fibre_efield = funcs.optimize_fibre(self.pupil, self.dx)

        return self.pupil

    def init_fftw(self):

        logger.info("Initialising FFTW")

        size = self.Niter_per_chunk
        if not self.temporal:
            size //= 2
            s = (size, *self.powerspec.shape)
        else:
            s = self.powerspec_per_layer.shape

        self.fftw_objs = {}
        self.fftw_objs['IN'] = pyfftw.empty_aligned(s, dtype='complex128')
        self.fftw_objs['OUT'] = pyfftw.empty_aligned(s, dtype='complex128')
        
        # NOTE: numpy and fftw have opposite exponents!
        self.fftw_objs['FFT'] = pyfftw.FFTW(self.fftw_objs['IN'], self.fftw_objs['OUT'], 
                                            axes=((-1,-2)),
                                            flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'],
                                            threads=self.nthreads) 

    def init_phs_logamp(self):
        logger.info("Initialising phase and log-amplitude arrays")
        self.phs = numpy.zeros((self.Niter_per_chunk, self.Npxls_pup, self.Npxls_pup))
        self.logamp = numpy.zeros((self.Niter))

    def compute_powerspec(self):
        logger.info("Computing (residual) phase power spectra")

        self.turb_powerspec = funcs.turb_powerspectrum_vonKarman(
            self.freq.main, self.cn2, self.L0, self.l0)

        self.G_ao = ao_power_spectra.G_AO_PAOLA(
            self.freq.main, self.lf_mask, self.ao_mode, self.h, 
            self.wind_vector, self.dtheta, self.D_ground, self.wvl, self.Zmax, 
            self.tloop, self.texp)

        self.aniso_servo_error = funcs.integrate_powerspectrum(funcs.integrate_path(
            self.G_ao * self.turb_powerspec, self.h, layer=True) * self.lf_mask * 2 * numpy.pi * self.k**2, self.freq.main.f)

        if self.alias and self.ao_mode != 'NOAO':
            self.alias_powerspec = ao_power_spectra.Jol_alias_openloop(
                self.freq.main, self.Dsubap, self.cn2, self.lf_mask, self.wind_vector,
                self.texp, self.wvl, 5, 5, self.L0, self.l0)

            self.alias_error = funcs.integrate_powerspectrum(funcs.integrate_path(
                self.alias_powerspec * 2 * numpy.pi * self.k**2, self.h, layer=True), self.freq.main.f)
        else:
            self.alias_powerspec = 0.
            self.alias_error = 0.

        if self.noise > 0 and self.ao_mode != 'NOAO':
            self.noise_powerspec = ao_power_spectra.Jol_noise_openloop(
                self.freq.main, self.Dsubap, self.noise, self.lf_mask)
            self.noise_error = funcs.integrate_powerspectrum(self.noise_powerspec, self.freq.main.f)
        else:
            self.noise_powerspec = 0.
            self.noise_error = 0.

        self.powerspec_per_layer = 2 * numpy.pi * self.k**2 \
            * (self.turb_powerspec * self.G_ao + self.alias_powerspec) + self.noise_powerspec / len(self.h)

        self.powerspec = funcs.integrate_path(self.powerspec_per_layer, h=self.h, layer=True)

        self.fitting_error = funcs.integrate_powerspectrum(self.powerspec * self.hf_mask, self.freq.main.f)
        self.phs_var = funcs.integrate_powerspectrum(self.powerspec, self.freq.main.f)
        self.phs_var_weights = funcs.integrate_powerspectrum(self.powerspec_per_layer, self.freq.main.f) / self.phs_var

        logger.info("Computing (residual) phase power spectra")

        # Log-amplitude powerspectrum
        self.logamp_powerspec = ao_power_spectra.logamp_powerspec(self.freq.main, 
            self.h, self.cn2, self.wvl, pupilfilter=self.pupil_filter, layer=True, L0=self.L0, l0=self.l0)
        self.logamp_var = funcs.integrate_powerspectrum(self.logamp_powerspec, self.freq.main.f)

        if self.subharmonics:
            logger.info("Computing subharmonics power spectra")
            self.turb_lo = funcs.turb_powerspectrum_vonKarman(
                self.freq.subharm, self.cn2, self.L0, self.l0)

            self.G_ao_lo = ao_power_spectra.G_AO_PAOLA(
                self.freq.subharm, self.lf_mask_subharm, 
                self.ao_mode, self.h, self.wind_vector, self.dtheta, self.D_ground, self.wvl, self.Zmax, 
                self.tloop, self.texp, self.Dsubap, self.modal, self.modal_mult)

            if self.alias and self.ao_mode != 'NOAO':
                self.alias_subharm = ao_power_spectra.Jol_alias_openloop(
                    self.freq.subharm, self.Dsubap, 
                    self.cn2, self.lf_mask_subharm, self.wind_vector, self.texp, self.wvl, 5, 5,
                    self.L0, self.l0)
            else:
                self.alias_subharm = 0.

            if self.noise > 0 and self.ao_mode != 'NOAO':
                self.noise_subharm = ao_power_spectra.Jol_noise_openloop(
                    self.freq.subharm,
                    self.Dsubap, self.noise, self.lf_mask_subharm)
            else:
                self.noise_subharm = 0.

            self.powerspec_subharm_per_layer = 2 * numpy.pi * self.k**2 * \
                (self.turb_lo * self.G_ao_lo + self.alias_subharm) \
                + self.noise_subharm / len(self.h)

            self.powerspec_subharm = funcs.integrate_path(self.powerspec_subharm_per_layer, h=self.h, layer=True) 
    
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

            logger.info("Computing temporal power spectra")

            logger.debug("Computing spatial shifts")
            dts = numpy.arange(1, self.Niter_per_chunk+1) * self.dt
            self.pixel_shifts = dts * self.wind_vector[...,numpy.newaxis] / self.dx

            # NOTE I think we have no need for phase temporal ps so have removed this calculation:

            # self.turb_temporal = funcs.turb_powerspectrum_vonKarman(
            #     self.freq.temporal, self.cn2, self.L0, self.l0)

            # self.G_ao_temporal = ao_power_spectra.G_AO_PAOLA(
            #     self.freq.temporal, self.lf_mask_temporal, 
            #     self.ao_mode, self.h, self.wind_vector, self.dtheta, self.D_ground, self.wvl, self.Zmax, 
            #     self.tloop, self.texp, self.Dsubap, self.modal, self.modal_mult)

            # if self.alias and self.ao_mode != 'NOAO':
            #     self.alias_temporal = ao_power_spectra.Jol_alias_openloop(
            #         self.freq.temporal, self.Dsubap, 
            #         self.cn2, self.lf_mask_temporal, self.wind_vector, self.texp, self.wvl, 10, 10,
            #         self.L0, self.l0)
            # else:
            #     self.alias_temporal = 0.

            # if self.noise > 0 and self.ao_mode != 'NOAO':
            #     noise_temporal = ao_power_spectra.Jol_noise_openloop(
            #         self.freq.temporal, self.Dsubap, self.noise, self.lf_mask_temporal)
            #     self.noise_temporal = funcs.integrate_path(noise_temporal, h=self.h, layer=True)
                
            # else:
            #     self.noise_temporal = 0.

            # temporal_powerspec_beforeintegration = 2 * numpy.pi * self.k**2 * \
            #     funcs.integrate_path(self.turb_temporal * self.G_ao_temporal + self.alias_temporal, h=self.h, layer=True) \
            #     + self.noise_temporal

            # integrate along y axis
            # self.temporal_powerspec = temporal_powerspec_beforeintegration.sum(-2) * self.freq.main.dfy
            # self.temporal_powerspec[len(self.temporal_powerspec)//2] = 0. # ensure the middle is 0!
            self.temporal_powerspec = None

            logger.debug("Generating 2D temporal logamp powerspec (before integration)")
            temporal_logamp_powerspec_beforeintegration = ao_power_spectra.logamp_powerspec(
                self.freq.temporal, self.h, self.cn2, self.wvl, pupilfilter=self.pupil_filter_temporal, 
                layer=True, L0=self.L0, l0=self.l0)

            logger.debug("Integrate 2D temporal logamp powerspec along axis orthogonal to wind direction ")
            self.temporal_logamp_powerspec = temporal_logamp_powerspec_beforeintegration.sum(-2) * self.freq.main.dfy

    def compute_phs(self, chunk=0):

        self.phs[:] = 0

        rand = funcs.generate_random_coefficients((self.Niter_per_chunk//2, *self.powerspec.shape))
        rand *= numpy.sqrt(self.powerspec)

        self.phs[:] = funcs.make_phase_fft(rand, self.freq.main.df, self.fftw, self.fftw_objs, double=True)[:,self.pup_coords[0],:][:,:,self.pup_coords[1]]

        if self.subharmonics:

            rand_lo = funcs.generate_random_coefficients((self.Niter_per_chunk//2, *self.powerspec_subharm.shape))
            rand_lo *= numpy.sqrt(self.powerspec_subharm)

            self.phs += funcs.make_phase_subharm(rand_lo, self.freq, self.Npxls, self.dx, double=True)[:,self.pup_coords[0],:][:,:,self.pup_coords[1]]

        return self.phs
    
    def compute_phs_temporal(self, chunk=0):

        if chunk == 0:
        
            rand = funcs.generate_random_coefficients(self.powerspec_per_layer.shape)
            rand *= numpy.sqrt(self.powerspec_per_layer)

            scrns = funcs.make_phase_fft(rand, self.freq.main.df, self.fftw, self.fftw_objs, double=False)
            self._interps = [RectBivariateSpline(numpy.arange(self.Npxls), numpy.arange(self.Npxls), i, kx=1, ky=1, s=0) for i in scrns]

            self.interp_coords = self.pup_coords[numpy.newaxis,:,numpy.newaxis,:].astype(float) + self.pixel_shifts[:,:,:,numpy.newaxis]

        self.phs[:] = 0

        coord = self.interp_coords % self.Npxls
        coord.sort(-1)

        diffs = numpy.abs(numpy.diff(coord, axis=-1))
        shifts = diffs.argmax(-1)
        shifts[numpy.isclose(diffs,1).all(-1)] = 0

        for i, scrn in enumerate(self._interps):
            for j in range(self.Niter_per_chunk):
                x = coord[i,0,j]
                y = coord[i,1,j]
                p = scrn(x, y)
                self.phs[j] += numpy.roll(p, -shifts[i,:,j], axis=(0,1))

        self.interp_coords += self.pixel_shifts[:,:,-1,numpy.newaxis, numpy.newaxis]

        return self.phs

    def compute_logamp(self):

        self.logamp[:] = 0
        self.logamp[:] = \
            funcs.generate_random_coefficients_logamp(self.Niter, self.logamp_var, self.temporal, self.temporal_logamp_powerspec).real

        return self.logamp

    def compute_detector(self, chunk=0):

        pupil = self.pupil * self.pupil_mode

        phase_component = (pupil * numpy.exp(1j * self.phs)).sum((1,2)) * self.dx**2
        logamp_component = numpy.exp(2*self.logamp[chunk*self.Niter_per_chunk:(chunk+1)*self.Niter_per_chunk])

        self.random_iters = logamp_component * phase_component
        normalisation = pupil.sum() * self.dx**2

        self.random_iters /= normalisation
            
        # if self.params['PROP_DIR'] == 'up':
        #     # Far field intensity
        #     self.I /= (self.wvl * self.L)
        #     self.diffraction_limit /= (self.wvl * self.L)

        if not self.params['COHERENT']:
            # incoherent detection (intensity)
            self.random_iters = numpy.abs(self.random_iters)**2

        return self.random_iters

    def compute_link_budget(self):
        '''
        Compute analytical losses/gains that affect the link. These are:

        power: Laser power expressed in [dBm]
        free_space: Losses due to free space propagation [dB]
        transmitter_gain: Gain due to transmitter [dBi]
        receiver_gain: Gain due to receiver [dBi]
        transmission_loss: Loss due to atmospheric transmission [dB]        
        smf_coupling: Losses due to coupling into single mode fibre [dB] (NOTE: this 
            refers to diffraction limited loss, does not include any 
            turbulence effects on the coupling)
    
        '''
        logger.info("Computing analytical link budget")

        if self.params['PROP_DIR'] == "up":
            D_t = self.D_ground
            D_r = self.D_sat
            obsc_t = self.obsc_ground
            obsc_r = self.obsc_sat
            mode = self.pupil_mode_sat
            dx_t = self.dx
            dx_r = self.dx_sat
            pupil_t = self.pupil
            pupil_r = self.pupil_sat
            w0 = self.W0
        else:
            D_t = self.D_sat
            D_r = self.D_ground
            obsc_t = self.obsc_sat
            obsc_r = self.obsc_ground
            mode = self.pupil_mode
            dx_t = self.dx_sat
            dx_r = self.dx
            pupil_t = self.pupil_sat
            pupil_r = self.pupil
            w0 = self.W0_sat

        self.link_budget = {}

        self.link_budget['power'] = 10*numpy.log10(self.power/1e-3)

        self.link_budget['free_space'] = 10*numpy.log10((self.wvl/(4*numpy.pi*self.L))**2)

        # eq 9, Klein et al Applied Optics 1974
        # TODO: make this work for axicon (bessel) beams
        alpha = D_t / w0
        gamma = obsc_t / D_t
        g_t = 2/alpha**2 * (numpy.exp(-alpha**2) - numpy.exp(-gamma**2 * alpha**2))**2
        G_t = 10*numpy.log10((numpy.pi * D_t**2) * 4*numpy.pi / self.wvl**2 * g_t)
        self.link_budget['transmitter_gain'] = G_t

        A = numpy.pi * ((D_r/2)**2 - (obsc_r/2)**2)
        G_r = 10*numpy.log10(4*numpy.pi*A / self.wvl**2)
        self.link_budget['receiver_gain'] = G_r

        self.link_budget['transmission_loss'] = 10*numpy.log10(self.params['TRANSMISSION'])

        smf_coupling = 10*numpy.log10(((pupil_r * mode).sum() * dx_r)**2 / (mode**2).sum())
        self.link_budget['smf_coupling'] = smf_coupling

        self.diffraction_limit = 10**(sum(self.link_budget.values())/10) / 1e3 # W

        return self.link_budget

    def compute_mean_irradiance(self, onaxis=True):
        '''
        FAST method using Fourier model (no Monte Carlo element)
        '''
        logger.info("Computing mean irradiance/coupled flux")

        pupil = numpy.zeros(self.powerspec.shape)
        pupil[:self.pupil.shape[0], :self.pupil.shape[1]] = self.pupil * self.pupil_mode

        phs_otf = fouriertransform.ift2(self.powerspec, self.freq.df)
        phs_sf = phs_otf[phs_otf.shape[0]//2, phs_otf.shape[1]//2] - phs_otf

        pupil_ft = fouriertransform.ft2(pupil, self.dx)
        pupil_otf = fouriertransform.ift2(numpy.abs(pupil_ft)**2, self.freq.df) / (2*numpy.pi)**2

        otf = numpy.exp(-phs_sf) * pupil_otf

        if not onaxis:
            psf = fouriertransform.ft2(otf, self.dx).real
        else:
            psf = otf.sum().real * self.dx**2

        normalisation = (pupil.sum() * self.dx**2)**2
        psf *= self.diffraction_limit / normalisation 

        return psf

    def calc_zenith_correction(self, zenith_angle):
        zenith_angle_rads = numpy.radians(zenith_angle)
        gamma = 1/numpy.cos(zenith_angle_rads)
        return gamma

    def set_seed(self, seed):
        funcs._R = numpy.random.default_rng(seed)

    def make_header(self, params):
        logger.info("Making FITS header")

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
        hdr['AO_MODE'] = self.ao_mode
        hdr['TLOOP'] = params['TLOOP'] 
        hdr['TEXP'] = params['TEXP']
        hdr['DSUBAP'] = params['DSUBAP'] 
        hdr['ALIAS'] = str(params['ALIAS'])
        hdr['NOISE'] = params['NOISE']
        hdr['D_GND'] = params['D_GROUND']
        hdr['OBSC_GND'] = params['OBSC_GROUND']
        hdr['D_SAT'] = params['D_SAT']
        hdr['OBSC_SAT'] = params['OBSC_SAT']
        hdr['AXICON'] = str(params['AXICON'])
        hdr['L_SAT'] = self.L
        hdr['H_SAT'] = params['H_SAT']
        hdr['DX'] = self.dx
        hdr['NPXLS'] = self.Npxls
        hdr['NITER'] = self.Niter
        hdr['R0'] = self.r0
        hdr['THETA0'] = self.theta0
        hdr['TAU0'] = self.tau0
        hdr["DIFFLIM"] = self.diffraction_limit
        if self.seed != None:
            hdr["SEED"] = self.seed
        return hdr

    def save(self, fname, **kwargs):
        logger.info(f"Saving results to {fname}")
        hdr = self.make_header(self.params) 
        fits.writeto(fname, self.result.power, header=hdr, **kwargs)

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
        fy_axes = []
        
        for i in range(nlayer):
            dx = wind_speed[i] * dt
            # df_temporal = 2 * numpy.pi / (Nx * dx)
            df_temporal = 1 / (Nx * dx) # NOTE Linear spatial frequency here!!!

            # define x axis according to temporal requirements, and y axis 
            # same as the main y axis, since we will integrate over this one
            fx_axis = numpy.arange(-Nx/2, Nx/2) * df_temporal
            fx_axes.append(fx_axis)

            fy_axis = numpy.arange(-Ny/2, Ny/2) * self.main.dfy
            fy_axes.append(fy_axis)

        self.temporal = SpatialFrequencyStruct(numpy.array(fx_axes), numpy.array(fy_axes), rot=numpy.radians(wind_dir), freq_per_layer=True)

    def make_logamp_freqs(self, Nx=None, dx=None, Ny=None, dy=None):
        if Nx is None and dx is None:
            self.logamp = self.main
           
        else:
            dfx = 2*numpy.pi/(Nx*dx)
            fx_axis = numpy.arange(-Nx/2., Nx/2.) * dfx
            dfy = 2*numpy.pi/(Ny*dy)
            fy_axis = numpy.arange(-Ny/2., Ny/2.) * dfy
            self.logamp = SpatialFrequencyStruct(fx_axis, fy_axis)

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
    

class FastResult():
    '''
    Allows rapid conversion between useful units for FAST results without having 
    to do all the conversions unless you need to.

    Attributes: 
        dB_rel: results in dB relative to diffraction limit (no turbulence)
        dB_abs: results in dB including all terms in link budget (i.e. received power/launched power)
        dBm: results in dBm, including all terms in link budget
        power: results in received power, units of Watts
        scintillation_index: scintillation index (variance of power/power.mean())
    '''
    def __init__(self, random_iters, diffraction_limit, header=None):
        self._r = random_iters
        self._dl = diffraction_limit
        if header != None:
            self.hdr = header
    
    @property
    def dB_rel(self):
        return 10*numpy.log10(self._r)
    
    @property 
    def dB_abs(self):
        return 10*numpy.log10(self._r * self._dl)
    
    @property
    def dBm(self):
        return 10*numpy.log10(self._r * self._dl / 1e-3)

    @property
    def power(self):
        return self._dl * self._r

    @property
    def scintillation_index(self):
        return (self._r/self._r.mean()).var()
    
    @property
    def avg_power_W(self):
        return self.power.mean() 

    @property
    def avg_power_dBm(self):
        return 10*numpy.log10(self.avg_power_W / 1e-3)
    
    @property
    def avg_power_dB_rel(self):
        return 10*numpy.log10((self.power / self._dl).mean())
    
    @property
    def avg_power_dB_abs(self):
        return 10*numpy.log10(self.avg_power_W)
    
    def __str__(self):
        s = \
        f"""FAST result statistics:
            Avg. power (W): {self.avg_power_W}
            Avg. power (dBm): {self.avg_power_dBm}
            Avg. power (dB_rel): {self.avg_power_dB_rel}
            Avg. power (dB_abs): {self.avg_power_dB_abs}
            Scintillation index: {self.scintillation_index}
        """
        return s

    

def load(fname):
    hdr = fits.getheader(fname)
    data = fits.getdata(fname)
    data /= hdr['DIFFLIM'] # assume saved in units of power 
    return FastResult(data, hdr['DIFFLIM'], header=hdr)