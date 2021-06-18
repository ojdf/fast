import numpy
from scipy.special import erfc
from scipy.integrate import simps
from scipy.optimize import minimize_scalar
from scipy.ndimage import rotate
from scipy.interpolate import RectBivariateSpline
from . import ao_power_spectra
from aotools import fouriertransform, circle, gaussian2d
import mpmath
import pyfftw
mp_kv = numpy.frompyfunc(mpmath.besselk, 2, 1)
mp_arraypower = numpy.frompyfunc(mpmath.power, 2, 1)

def f_grid_linear(L0, l0, max_size=1024):
    '''
    Spatial frequency grid with linear spacing.

    Parameters
        L0 (float): Outer scale. Defines the minimum spatial frequency
        l0 (float): Inner scale. Defines the maximum spatial frequency, assuming 
            that it will result in an array smaller than max_size
        max_size (int): Maximum array size, will override inner scale 
        
    Returns
        fx (numpy.ndarray): x spatial frequencies
        fy (numpy.ndarray): y spatial frequencies
        fabs (numpy.ndarray): |f| = sqrt(fx**2 + fy**2)
        f (numpy.ndarray): 1d spatial frequencies array
    '''
    df = 2 * numpy.pi / L0
    fmax = 2 * 5.92 / l0
    N = 2 * fmax/df + 1
    if N > max_size:
        fmax = max_size * df / 2
    f = numpy.arange(-fmax, fmax, df)

    fx, fy = numpy.meshgrid(f,f)
    return fx, fy, numpy.sqrt(fx**2 + fy**2), f

def f_grid_dx(N, dx):
    '''
    Spatial frequency grid given real-space grid spacing and size.

    Parameters
        N (int): Input array size (along one axis)
        dx (float): Input array spacing (m/pixel)

    Returns
        fx (numpy.ndarray): x spatial frequencies
        fy (numpy.ndarray): y spatial frequencies
        fabs (numpy.ndarray): |f| = sqrt(fx**2 + fy**2)
        f (numpy.ndarray): 1d spatial frequencies array
    '''
    df = 2*numpy.pi/(N*dx)
    fx = numpy.arange(-N/2., N/2.) * df
    (fx, fy) = numpy.meshgrid(fx,fx)
    fabs = numpy.sqrt(fx**2 + fy**2)
    return fx, fy, fabs, fx[0]

def f_grid_log(L0, l0, N=129, include_0=True):
    '''
    Spatial frequency grid with logarithmic spacing.

    Parameters
        L0 (float): Outer scale. Defines the minimum spatial frequency
        l0 (float): Inner scale. Defines the maximum spatial frequency
        N (int): Number of points along one side.

    Returns
        fx (numpy.ndarray): x spatial frequencies
        fy (numpy.ndarray): y spatial frequencies
        fabs (numpy.ndarray): |f| = sqrt(fx**2 + fy**2)
        f (numpy.ndarray): 1d spatial frequencies array
    '''
    if N%2 == 0:
        N_one_side = int(N/2)
    else:
        N_one_side = int((N-1)/2)

    fmin = 0.5 * (2 * numpy.pi) / L0
    fmax = 2 * (2 * numpy.pi) / l0

    f_one_side = numpy.logspace(numpy.log10(fmin), numpy.log10(fmax), N_one_side)
    if include_0:
        f = numpy.hstack([-f_one_side[::-1], 0, f_one_side])
    else:
        f = numpy.hstack([-f_one_side[::-1], f_one_side])
    fx, fy = numpy.meshgrid(f,f)
    return fx, fy, numpy.sqrt(fx**2 + fy**2), f

def integrate_powerspectrum(power_spectrum, f):
    '''
    Integrate a power spectrum or cube of power spectra using simpsons rule 
    numerical integration. Integration is performed over the last two axes of 
    the input.

    Parameters
        power_spectrum (numpy.ndarray): 2D or 3D array of power spectra
        f (numpy.ndarray): Spatial frequency axis (along one side, assumed same for 
            both axes)
        
    Returns
        integral (numpy.ndarray): Power spectrum integrated across final two axes 
            of the input
    '''
    return simps(simps(power_spectrum, x=f), x=f)

def integrate_path(integrands, h, layer=False, axis=0):
    '''
    Integrate along the path (i.e. height). Takes into account if the atmosphere 
    is using a discrete layered model with cn2 dh values or a continuous model 
    with cn2 values.

    Parameters
        integrands (numpy.ndarray): 1D array of integrated power spectra.
        h (numpy.ndarray): Height of each layer
        layer (bool): Whether a discrete layer model is used or not. Default is False.
        axis (int): Axis over which to integrate

    Returns
        integral (numpy.ndarray): Integral of input array over the 0 axis
    '''
    if layer:
        # Integration is just a sum in this case
        return integrands.sum(axis)
    else:
        return simps(integrands, x=h, axis=axis)

def turb_powerspectrum_vonKarman(freq, cn2, L0=25, l0=0.01, C=2*numpy.pi):
    '''
    Von Karman turbulence power spectrum of refractive index.

    Parameters
        freq (SpatialFrequencyStruct): spatial frequency object from sim
        cn2[dh] (float / numpy.ndarray): Refractive index structure constant,
            can be a 1d array for multiple layers. If discrete layers are used 
            then this may be considered the cn2dh value, too.
        L0 (float): Outer scale
        l0 (float): Inner scale
        C (float): Multiplier for outer scale, i.e. k0=C/L0. Usually either 1 or 2pi
    '''
    fabs = freq.fabs
    km = 5.92 / l0
    k0 = C / L0
    try:
        nlayers = len(cn2)
        cn2 = numpy.array(cn2)
        if freq.freq_per_layer:
            # we have a 3d fabs array already, with an entry for each layer
            power_spec = ((0.033 * numpy.exp(-fabs**2/km**2) / (fabs**2 + k0**2)**(11/6.)).T * cn2).T
        else:
            power_spec = (numpy.array([0.033 * numpy.exp(-fabs**2/km**2) / (fabs**2 + k0**2)**(11/6.)]*nlayers).T * cn2).T 
    except TypeError:
        # cn2 is a single float value
        power_spec = numpy.array([0.033 * cn2 * numpy.exp(-fabs**2/km**2) / (fabs**2 + k0**2)**(11/6.) ])

    # Set any infinite values to 0
    power_spec[numpy.isinf(power_spec)] = 0.
    # if numpy.isinf(L0) and fabs[midpt,midpt] == 0:
    #     power_spec[:,int(power_spec.shape[1]/2), int(power_spec.shape[2]/2)] = 0.
    return power_spec 

def calc_gaussian_beam_parameters(z, F_0, W_0, wvl):
    '''
    Andrews and Phillips Chapter 12 eq. 8 and 9

    Parameters
        z (float): Propagation distance 
        F_0 (float): Phase front radius of curvature at transmitter
        W_0 (float): 1/e beam radius at transmitter
        wvl (float): Wavelength 

    Returns
        Theta_0, Lambda_0: Input plane beam parameters
        Theta, Lambda, Theta_bar: Output plane beam parameters
    '''
    k = 2 * numpy.pi / wvl
    Theta_0 = 1 - z/F_0
    Lambda_0 = 2*z / (k * W_0**2)
    Theta = Theta_0 / (Theta_0**2 + Lambda_0**2)
    Theta_bar = 1 - Theta
    Lambda = Lambda_0 / (Theta_0**2 + Lambda_0**2)
    return Theta_0, Lambda_0, Theta, Lambda, Theta_bar

def pdf_lognorm(Is, sigma, Imn=1):
    scint = sigma**2
    pdf = 1/(Is * numpy.sqrt(scint * 2*numpy.pi)) * numpy.exp(-(numpy.log(Is/Imn) + 0.5 * scint)**2 / (2*scint))
    return pdf

def pdf_gammagamma(Is, alpha, beta):
    pI_1 = 2 * mpmath.power(alpha * beta, 0.5 * (alpha + beta))
    pI_2 = mpmath.gamma(alpha) * mpmath.gamma(beta)
    pI_3 = mp_arraypower(Is, (0.5 * (alpha+beta) - 1)) * mp_kv(alpha-beta, 2 * numpy.sqrt(alpha * beta * Is))
    pI = pI_1 * pI_3 / pI_2

    return pI

def fade_prob(Is_rand, threshold):
    I_thres = threshold * Is_rand.mean()
    return (Is_rand<I_thres).sum()/len(Is_rand)

def BER_ook(Is_rand, SNR, bins=None, nbins=100):
    if bins is None:
        bins = numpy.logspace(numpy.log10(Is_rand.min()), numpy.log10(Is_rand.max()), nbins)

    weights, edges = numpy.histogram(Is_rand, bins=bins, density=True)

    centres = edges[:-1] + numpy.diff(edges)/2.

    integrand = 0.5 * weights * erfc(SNR * centres/(2*numpy.sqrt(2)))

    integral = simps(integrand, x=centres)

    return integral

def make_phase_fft(Nscrns, freq, powerspec, sh=False, powerspecs_lo=None, dx=None, 
                    fftw=False, temporal=False, temporal_powerspec=None):

    df = freq.df

    rand = generate_random_coefficients(Nscrns, powerspec, 
                temporal=temporal, temporal_powerspecs=temporal_powerspec)

    if fftw:
        fftw_in = pyfftw.empty_aligned(rand.shape, dtype='complex128')
        fftw_out = pyfftw.empty_aligned(rand.shape, dtype='complex128')
        fftw_obj = pyfftw.FFTW(fftw_in, fftw_out, axes=((-1,-2)))
        fftw_in[:] = numpy.fft.fftshift(rand * df, axes=(-1,-2))
        phasescrn = numpy.fft.fftshift(fftw_obj(), axes=(-1,-2)).real

    else:
        phasescrn = fouriertransform.ft2(rand*df, 1).real

    if sh:
        # subharmonics

        N = phasescrn.shape[-1]
        phs_lo = numpy.zeros(phasescrn.shape[1:])
        D = dx * N
        coords = numpy.arange(-D/2, D/2, dx)
        if len(coords) == N+1:
            coords = coords[:-1]
        x, y = numpy.meshgrid(coords, coords)

        # preallocate data arrays
        SH = numpy.empty((Nscrns,N,N), dtype='complex')
        tmp = numpy.empty((Nscrns,3,3,N,N), dtype='complex')

        for i,p in enumerate(range(1,4)):
            df_lo = 2*numpy.pi/(3**p * D)
            fx_lo = freq.subharm.fx[i]
            fy_lo = freq.subharm.fy[i]
            fabs_lo = freq.subharm.fabs[i]

            powerspec_lo = powerspecs_lo[i]

            powerspec_lo[1,1] = 0

            rand_lo = generate_random_coefficients(Nscrns, powerspec_lo,
                        temporal=temporal, temporal_powerspecs=temporal_powerspec) \
                            * df_lo

            modes = numpy.exp(1j * (x[numpy.newaxis,numpy.newaxis,...] * fx_lo[...,numpy.newaxis,numpy.newaxis]
                                  + y[numpy.newaxis,numpy.newaxis,...] * fy_lo[...,numpy.newaxis,numpy.newaxis]))

            numpy.multiply(rand_lo[...,numpy.newaxis,numpy.newaxis], modes, out=tmp)
            numpy.sum(tmp, axis=(1,2), out=SH)

            # old, slightly slower method
            # SH = numpy.zeros((Nscrns,N,N), dtype='complex')
            # for i in range(3):
            #     for j in range(3):
            #         SH += (rand_lo[:,i,j] * numpy.tile(numpy.exp(1j * (fx_lo[i,j]*x + fy_lo[i,j]*y)), (Nscrns,1,1)).T).T

            phs_lo = phs_lo + SH

        phs_lo = (phs_lo.real.T - phs_lo.real.mean((1,2))).T

    else:
        phs_lo = 0
    
    phs = phasescrn + phs_lo

    return phs
    
def logamp_var(pupil, freq, dx, h, cn2, wvl, L0=numpy.inf, l0=1e-6):
    N = pupil.shape[-1]

    fabs_3d = numpy.array([freq.fabs]*len(h))

    powerspec = turb_powerspectrum_vonKarman(freq, cn2, L0=L0, l0=l0) * 2*numpy.pi*(2*numpy.pi/wvl)**2

    P =  numpy.abs(fouriertransform.ft2(pupil, dx))**2
    P /= (pupil.sum() * dx**2)**2

    integrand = (powerspec * numpy.sin(wvl * h * fabs_3d.T**2 / (4 * numpy.pi)).T**2).sum(0) \
        * P

    return integrate_powerspectrum(integrand, freq.f)

def compute_pupil(N, dx, Tx, W0=None, Tx_obsc=0, ptype='gauss'):
    circ_ap = circle(Tx/dx/2, N) - circle(Tx_obsc/dx/2, N)

    if ptype is 'circ':
        return circ_ap / numpy.sqrt(circ_ap.sum()*dx**2)

    elif ptype is 'gauss':
        I0 = 2 / (numpy.pi * W0**2)
        return gaussian2d(N, W0/dx/numpy.sqrt(2)) * circ_ap * numpy.sqrt(I0)

    elif ptype is 'axicon':
        x = numpy.arange(-N/2, N/2, 1) * dx
        xx, yy = numpy.meshgrid(x,x)
        r = numpy.sqrt(xx**2 + yy**2)
        midpt = Tx_obsc/2 + (Tx/2-Tx_obsc/2)/2
        ring = numpy.exp(-(r - midpt)**2 / W0**2) 
        P = (ring**2).sum() * dx**2 
        return ring * circ_ap / numpy.sqrt(P)

    else:
        raise Exception('ptype must be one of "circ", "gauss" or "axicon"')

def pupil_filter(freq, pupil, spline=False):
    P = numpy.abs(fouriertransform.ft2(pupil, 1))**2
    P /= pupil.sum()**2

    if spline:
        return RectBivariateSpline(freq.fx_axis, freq.fy_axis, P, kx=1, ky=1, s=0)
    else:
        return P

def optimize_fibre(pupil, dx, size_min=None, size_max=None):
    N = pupil.shape[-1]

    if size_max is None:
        size_max = N * dx

    if size_min is None:
        size_min = dx

    def _opt_func(W):
        return coupling_loss(W, N, pupil, dx)

    opt = minimize_scalar(_opt_func, bracket=[size_min, size_max]).x
    
    return gaussian2d(N, opt/dx/numpy.sqrt(2)) * numpy.sqrt(2./(numpy.pi * opt**2))

def coupling_loss(W, N, pupil, dx):
    fibre_field = gaussian2d(N, W/dx/numpy.sqrt(2)) * numpy.sqrt(2./(numpy.pi*W**2))
    coupling = numpy.abs((fibre_field * pupil).sum() * dx**2)**2
    return 1 - coupling

def generate_random_coefficients(Nscrns, powerspec,  temporal=False, temporal_powerspecs=None):

    if not temporal:

        rand = numpy.random.normal(0,1,size=(Nscrns, *powerspec.shape)) \
         + 1j * numpy.random.normal(0,1,size=(Nscrns, *powerspec.shape))

        return rand * numpy.sqrt(powerspec)

    else:

        r_fourier = numpy.random.normal(0,1,size=(*powerspec.shape, Nscrns)) \
            + 1j * numpy.random.normal(0,1,size=(*powerspec.shape, Nscrns))

        r_fourier *= numpy.sqrt(temporal_powerspecs/temporal_powerspecs.sum())

        r = fouriertransform.ft(r_fourier,1)

        return r.T * numpy.sqrt(powerspec)

def temporal_autocorrelation(I):
    # normalise
    I -= I.mean()
    I /= I.std()

    corr = numpy.correlate(I,I, mode='full')

    return corr[len(I):] / len(I)

