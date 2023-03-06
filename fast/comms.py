'''
Functions regarding optical communications
'''
import numpy
from . import Fast
from scipy.special import erfc 
from scipy.ndimage import correlate1d

class Modulator():
    '''
    Takes series of intensities (or coherent field values) and modulates/demodulates 
    according to a modulation scheme (OOK, BPSK, QPSK, QAM, etc) with random bits. 
    This allows Monte Carlo computation of bit error rate or symbol error probability.  

    Parameters:
        I (numpy.ndarray): Array of intensities/coherent field values 
        modulation (string): Modulation scheme 
        N0 (float, optional): Noise variance for AWGN
    '''
    def __init__(self, I, modulation, N0=0):
        self.I = I
        self.modulation = modulation
        self.N0 = N0

    def generate_symbols(self):

        if self.modulation in ['OOK', 'BPSK']:
            self.nsymbols = 2

        elif self.modulation in ["QPSK", "QAM"]:
            self.nsymbols = 4
            
        elif len(self.modulation.split("-")) ==2:
            self.nsymbols = int(self.modulation.split("-")[0])

        else:
            raise ValueError("Scheme not recognised")

        self.symbols = numpy.random.randint(0, self.nsymbols, size=len(self.I))

    def modulate(self):

        if self.modulation == None:
            self.recv_signal = self.I
            return self.recv_signal

        self.generate_symbols()

        if self.N0 > 0:
            # AWGN (generate complex, if incoherent then take abs)
            self.awgn = numpy.random.normal(0, self.N0/numpy.sqrt(2), size=len(self.I)) \
                + 1j * numpy.random.normal(0, self.N0/numpy.sqrt(2), size=len(self.I))
        else:
            self.awgn = 0
    
        # incoherent on-off keying
        if self.modulation == "OOK":
            if self.params['COHERENT']:
                raise ValueError(f"{self.modulation} modulation requires COHERENT=False")

            self.recv_signal = self.symbols * self.I + numpy.abs(self.awgn)**2
            self.Es = (self.symbols * self.I).mean()
            return self.recv_signal

        # coherent schemes
        if self.I.dtype != complex:
            raise ValueError(f"{self.modulation} modulation requires COHERENT=True")

        constellation = define_constellation(self.modulation)
        mod = constellation[self.symbols]

        # Constellation normalised by mean amplitude
        self.constellation = abs(self.I).mean() * constellation

        # calculate average symbol energy Es (useful later)
        self.Es = (numpy.abs(self.constellation)**2).mean()

        # Received signal loses any atmospheric phase aberrations because of 
        # phase locked loop on reciever?
        self.recv_signal = mod * abs(self.I) + self.awgn
        
        return self.recv_signal

    def demodulate(self):
        
        if self.modulation == None:
            self.recv_symbols = None
            return self.recv_symbols
        
        # fast ways for OOK and BPSK
        if self.modulation == "OOK":
            cutoff = 0.5 * self.I.mean()
            self.recv_symbols = (self.recv_signal > cutoff).astype(int)

        elif self.modulation == "BPSK":
            self.recv_symbols = (self.recv_signal.real < 0).astype(int)
            
        else:
            d = numpy.array([abs(self.recv_signal - i) for i in self.constellation])
            self.recv_symbols = d.argmin(0)

        return self.recv_symbols

    def compute_sep(self):
        '''
        Symbol error probability, from random bits
        '''

        if self.modulation == None:
            self.sep = None

        else:
            self.sep = (self.recv_symbols != self.symbols).sum() / len(self.symbols)
            
        return self.sep

    def compute_evm(self):
        '''
        Error Vector Magnitude (EVM), from random bits
        '''

        if self.modulation == None:
            self.evm = None
            
        else:
            tx_signal = self.constellation[self.symbols]
            ref = numpy.sqrt((tx_signal.real**2 + tx_signal.imag**2).mean()) # RMS of constellation points
            self.evm = (abs(tx_signal - self.recv_signal) / ref).mean()
            
        return self.evm

    def run(self):
        self.modulate()
        self.demodulate()
        self.compute_sep()
        self.compute_evm()


class FastFSOC(Fast):
    '''
    Subclass of Fast simulation object, adds optical comms functionality 
    (modulation, demodulation, generating random symbol sequences for testing)
    '''

    def __init__(self, *args, **kwargs):
        super(FastFSOC, self).__init__(*args, **kwargs)
        self.modulation = self.params['MODULATION']
        self.N0 = self.params['N0']

    def run(self):
        super(FastFSOC, self).run()
        self.modulator = Modulator(self.I, self.modulation, self.N0)
        self.modulator.run()

    def make_header(self, params):
        hdr = super(FastFSOC, self).make_header(params)
        hdr['MODULATION'] = params['MODULATON']
        return hdr


def fade_prob(I, threshold, min_fades=30):
    prob = (I<threshold).sum()/len(I)
    if (I<threshold).sum() < min_fades:
        # not enough fades
        return numpy.nan
    else:
        return (I<threshold).sum()/len(I)


def fade_dur(I, threshold, dt=1, min_fades=30):
    fade_mask = I < threshold
    fade_start = numpy.where(numpy.diff(fade_mask.astype(int)) == 1)[0] + 1
    fades = numpy.array_split(fade_mask, fade_start)[1:]

    # select only fades that end within the window (i.e. final element is not fading)
    fades_filt = [i for i in fades if i[-1] != True]

    if len(fades_filt) < min_fades:
        # not enough fades to characterise duration, return nan
        return numpy.nan, numpy.nan

    fade_durs = [i.sum() for i in fades_filt]
    mn = numpy.mean(fade_durs)
    return mn * dt


def BER_ook(Is_rand, SNR, bins=None, nbins=100):
    if bins is None:
        bins = numpy.logspace(numpy.log10(Is_rand.min()), numpy.log10(Is_rand.max()), nbins)

    weights, edges = numpy.histogram(Is_rand, bins=bins, density=True)

    centres = edges[:-1] + numpy.diff(edges)/2.

    integrand = 0.5 * weights * erfc(SNR * centres/(2*numpy.sqrt(2)))

    integral = simps(integrand, x=centres)

    return integral


def sep_qam(samples, M, npxls, EsN0, N0=None):
    return 1-convolve_awgn_qam(samples, M, npxls, EsN0, N0=N0, region_size="individual").sum((-1,-2)).mean()


def mutual_information_qam(samples, M, npxls, EsN0, N0=None):
    '''
    Equation 16 from Alvarado et al (2016) 10.1109/JLT.2015.2450537.

    This is for a memoryless receiver (no knowledge of other bits transmitted)
    '''
    fyx = convolve_awgn_qam(samples, M, npxls, EsN0, N0=N0, region_size="full")

    fy = fyx.mean(0)
    return (fyx * (numpy.ma.log2(fyx)-numpy.ma.log2(fy))).sum((-1,-2)).mean()


def mi_old(samples, M, npxls, EsN0, N0=None):
    fyx =convolve_awgn_qam(samples, M, npxls, EsN0, N0=N0, region_size="full") 
    fy = fyx.mean(0)
    return numpy.nan_to_num(fyx * (numpy.log2(fyx)-numpy.log2(fy))).sum((-1,-2)).mean()


def convolve_awgn_qam(samples, M, npxls, EsN0, N0=None, region_size="individual"):
    '''
    Method of computing received I-Q plane for M-ary QAM under AWGN assumptions
    given a series of complex field measurements. 

    Bins the Monte Carlo field measurements into a 2D array of npxls x npxls bins. 
    The overall size of the 2d array is determined by the decision region size, 
    which depends on the M-ary QAM constellation. This is then convolved with a 
    gaussian to include AWGN. 

    By integrating this distribution, and averaging over all constellation regions, 
    we can obtain the probability that a transmitted symbol ends up outside the 
    decision region, i.e. a symbol error. This [may be] more robust than simply 
    adding AWGN to the individual Monte Carlo datapoints, because that method 
    is limited by the number of samples.

    Args:
        samples (numpy.ndarray): Array of Monte Carlo complex field measurements or amplitudes
        M (int): number of symbols (must be perfect square)
        npxls (int): Number of pixels to use for binning 
        EsN0 (float): Symbol signal-to-noise ratio [dB]
        N0 (float, optional): Noise variance (overrides EsN0, can be set to 0)
        separate (bool, optional): Separate each region of the constellation (default: True)
        region_size (str, optional): "individual" or "full" region to define the PDF (default: "individual")

    Returns:
        out (numpy.ndarray): (nsymbols x npxls x npxls) array consisting of the 
            binned histogram for each symbol. 
    '''
    # define constellation
    constellation = define_constellation(f"{M}-QAM")
    if region_size == "individual":
        decision_region_size = 1/(numpy.sqrt(M)-1) # this works and I don't know why
    elif region_size == "full":
        decision_region_size = 2 # slightly oversize
    else:
        raise ValueError("decision_region_size must be either 'full' or 'individual'")

    # normalise constellation to mean amplitude?
    constellation_norm = constellation * numpy.mean(numpy.abs(samples))
    decision_region_size_norm = decision_region_size * numpy.mean(numpy.abs(samples))

    # compute noise from desired SNR per symbol (EsN0) if required
    if N0 == None:
        Es = numpy.mean(numpy.abs(constellation_norm)**2)
        N0 = Es / 10**(EsN0/10)
    
    # for large variance, increase the size of decision_region_size for "full" 
    # type region to include +2sigma and avoid clipping the calculated PDF
    if region_size == "full":
        region_size_required = 2*(numpy.mean(numpy.abs(samples))/numpy.sqrt(2) + 2 * numpy.sqrt(N0))
        if region_size_required > decision_region_size_norm:
            print("AWGN noise level too large for region, increasing region size")
            # npxls = int(numpy.round(npxls * region_size_required / decision_region_size_norm))
            decision_region_size_norm = region_size_required 

    dx = decision_region_size_norm / npxls
    x_g = numpy.linspace(-npxls/2, npxls/2, npxls+1) 
    
    # compute variance in pixel units. if less than one, set equal to one to avoid 
    # numerical issues with normalisation
    sigma2 = N0 / (2 * dx**2)
    if sigma2 < 1:
        sigma2 = 1

    g = numpy.exp(-x_g**2 / sigma2) / numpy.sqrt(numpy.pi * sigma2)

    out = numpy.zeros((len(constellation), npxls, npxls))

    for c in range(len(constellation)):
        x = numpy.linspace(-decision_region_size_norm/2,decision_region_size_norm/2,npxls+1)
        y = numpy.linspace(-decision_region_size_norm/2,decision_region_size_norm/2,npxls+1)

        if region_size == "individual":
            x += constellation_norm[c].real
            y += constellation_norm[c].imag

        samples_norm = constellation[c] * numpy.abs(samples)
        h = numpy.histogram2d(samples_norm.real, samples_norm.imag, bins=[x,y])[0] / len(samples_norm)

        # perform separated 2d gaussian filter
        h = correlate1d(h, g, mode='constant', axis=0)
        h = correlate1d(h, g, mode='constant', axis=1)

        out[c] = h

    return out


# def _estimate_region_pad(sigma):
#     corner = 
    


def define_constellation(modulation):
    '''
    Define constellations for coherent modulation schemes. Schemes supported:

    BPSK - Binary Phase Shift Keying
    QPSK - Quadrature Phase Shift Keying 
    QAM - Quadrature Amplitude Modulation 
    M-PSK - M-ary PSK (e.g. 16-PSK)
    M-QAM - M-ary QAM (e.g. 16-QAM)

    Parameters:
        modulation (string): Modulation scheme (e.g. "BPSK" or "64-QAM")

    Returns:
        constellation (numpy.ndarray): Complex array representing the constellation, 
            of dimension (nsymbols), real and imaginary parts correspond to the
            two axes of the modulation.
    '''
    # coherent schemes
    if modulation == "BPSK":
        # binary phase shift keying
        nsymbols = 2
        constellation = numpy.exp(1j * numpy.arange(nsymbols) * numpy.pi)
    
    elif modulation in ["QPSK", "QAM"]:
        # quadrature PSK, quadrature amplitude modulation (same thing?)
        nsymbols = 4
        constellation = numpy.exp(1j * ((numpy.arange(nsymbols) * numpy.pi/2) - numpy.pi/4))

    elif (modulation[-4:] == "-PSK"):
        # M-PSK
        nsymbols = int(modulation[:-4])
        constellation = numpy.exp(1j * (numpy.arange(nsymbols) * numpy.pi/(nsymbols/2)))

    elif (modulation[-4:] == "-QAM"):
        # M-QAM
        # first, check that nsymbols is a perfect square (not bothering with non-square)
        nsymbols = int(modulation[:-4])
        if not (numpy.sqrt(nsymbols) == numpy.ceil(numpy.sqrt(nsymbols))):
            raise ValueError(f"{nsymbols}-QAM not possible as {nsymbols} is not a perfect square, only square M-QAM modulations supported")

        n_side = int(numpy.sqrt(nsymbols))
        x = numpy.linspace(-1,1,n_side) / numpy.sqrt(2)
        xx, yy = numpy.meshgrid(x,x)
        pos = (xx + 1j * yy).flatten()

        constellation = pos

    else:
        raise ValueError(f"Modulation scheme {modulation} not supported")

    return constellation


def Q(x):
    '''
    Q function from Rice book
    '''
    return 1/2 * erfc(x/numpy.sqrt(2))


def sep_qam_analytic(M, EsN0):
    '''
    Symbol error probabilty for square M-ary QAM, from Rice

    Parameters:
        M (int): Number of symbols (must be perfect square)
        EsN0 (float): Symbol signal-to-noise ratio [dB]
    '''
    EsN0_frac = 10**(EsN0/10)
    return 4*(numpy.sqrt(M)-1)/numpy.sqrt(M) * Q(numpy.sqrt(3/(M-1) * EsN0_frac) )


def bep_qam_analytic(M, EbN0):
    '''
    Bit error probability (rate) for square M-ary QAM, from Rice

    Parameters:
        M (int): Number of symbols (must be perfect square)
        EbN0 (float): Bit signal-to-noise ratio [dB]
    '''
    return 1/numpy.log2(M) * sep_qam(M, 10*numpy.log10(numpy.log2(M)) + EbN0)


def mutual_information_awgn_analytic(M, EsN0, nrand=10000):
    '''
    Eq. 28 from Alvarado et al (2016) 10.1109/JLT.2015.2450537.
    '''
    constellation = define_constellation(f"{M}-QAM")
    Es = numpy.mean(numpy.abs(constellation)**2)
    snr_linear = 10**(EsN0/10)
    N0 = Es / snr_linear
    r = numpy.random.normal(0,N0/numpy.sqrt(2),size=nrand) + 1j *numpy.random.normal(0,N0/numpy.sqrt(2),size=nrand)
    rabs = numpy.abs(r)**2
    snr_linear = Es / N0

    f = [[numpy.exp(-snr_linear * (2*(numpy.conjugate(xi - xj) * r).real + rabs))
            for xi in constellation] for xj in constellation]
    f = numpy.array(f).sum(0)
    f = numpy.log2(f)

    return numpy.log2(M) - f.mean()
        