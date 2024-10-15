'''
Functions regarding optical communications
'''
import numpy
from . import Fast
from scipy.special import erfc
from scipy.ndimage import correlate1d
from aotools import gaussian2d
import logging

logger = logging.getLogger(__name__)

class Modulator():
    '''
    Takes array of optical powers and modulates/demodulates according to a modulation
    scheme (OOK, BPSK, QPSK, QAM, etc) with random bits. Can add AWGN at a given
    average signal-to-noise ratio.

    This allows Monte Carlo computation of bit error rate or symbol error probability.

    Parameters:
        power (numpy.ndarray): array of optical powers
        modulation (string): modulation scheme
        EsN0 (float, optional): (average) symbol signal to noise ratio
        symbols_per_iter (int, optional): Number of symbols per iteration of FAST.
            Defaults to 1000.
    '''
    def __init__(self, power, modulation, EsN0=None, symbols_per_iter=1000, data=None):
        self.power = power / power.mean()
        self.amplitude = numpy.sqrt(self.power)
        self.modulation = modulation
        self.symbols_per_iter = symbols_per_iter
        self.EsN0 = EsN0
        self.data = data
        if EsN0 != None:
            self.snr = numpy.sqrt(10**(EsN0/10)) * self.power

    def generate_symbols(self):

        if self.modulation in ['OOK', 'BPSK']:
            self.nsymbols = 2

        elif self.modulation in ["QPSK", "QAM"]:
            self.nsymbols = 4

        elif len(self.modulation.split("-")) ==2:
            self.nsymbols = int(self.modulation.split("-")[0])

        else:
            raise ValueError("Scheme not recognised")

        self.bits_per_symbol = numpy.log2(self.nsymbols).astype(int)

        if self.data is not None:
            s, self._pad_bits = _encode(self.data, self.bits_per_symbol)
            self.symbols = numpy.array([s]*len(self.power)).T
            self.symbols_per_iter = len(s)
        else:
            self.symbols = numpy.random.randint(0, self.nsymbols, size=(self.symbols_per_iter, len(self.power)))

    def modulate(self):

        if self.modulation == None:
            self.recv_signal = self.power
            return self.recv_signal

        self.generate_symbols()

        self.constellation = define_constellation(self.modulation)
        mod = self.constellation[self.symbols]

        # calculate average symbol energy Es (useful later)
        self.Es = (numpy.abs(self.constellation)**2).mean()

        if self.EsN0 != None:
            if self.modulation == "OOK":
                self.awgn = numpy.random.normal(0, self.Es/self.snr, size=(self.symbols_per_iter, len(self.power)))
            else:
                self.awgn = numpy.random.normal(0, numpy.sqrt(self.Es/2)/self.snr, size=(self.symbols_per_iter,len(self.power))) \
                    + 1j * numpy.random.normal(0, numpy.sqrt(self.Es/2)/self.snr, size=(self.symbols_per_iter, len(self.power)))
        else:
            self.awgn = 0

        self.recv_signal = mod + self.awgn

        return self.recv_signal

    def demodulate(self):

        if self.modulation == None:
            self.recv_symbols = None
            return self.recv_symbols

        # fast ways for OOK and BPSK
        if self.modulation == "OOK":
            cutoff = 0.5
            self.recv_symbols = (self.recv_signal > cutoff).astype(int)

        elif self.modulation == "BPSK":
            self.recv_symbols = (self.recv_signal.real < 0).astype(int)

        else:
            d = numpy.array([abs(self.recv_signal - i) for i in self.constellation])
            self.recv_symbols = d.argmin(0)

        if self.data is not None:
            self.recv_data = numpy.zeros((len(self.power), self.symbols_per_iter), dtype=numpy.uint8)
            for i in range(self.symbols_per_iter):
                self.recv_data[i] = _decode(self.recv_symbols[:,i], self.bits_per_symbol, self._pad_bits)

        return self.recv_symbols

    def compute_sep(self):
        '''
        Symbol error probability, from random bits
        '''

        if self.modulation == None:
            self.sep = None

        else:
            self.sep = (self.recv_symbols != self.symbols).mean()

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
        self.EsN0 = self.params['EsN0']

    def run(self):
        super(FastFSOC, self).run()
        self.modulator = Modulator(self.result.power, self.modulation, self.EsN0)
        self.modulator.run()

    def make_header(self, params):
        hdr = super(FastFSOC, self).make_header(params)
        hdr['MODULATION'] = params['MODULATON']
        hdr['EsN0'] = self.EsN0
        return hdr


def fade_prob(I, threshold, min_fades=30):
    prob = (I<threshold).sum()/len(I)
    if (I<threshold).sum() < min_fades:
        # not enough fades
        return numpy.nan
    else:
        return prob


def fade_dur(I, threshold, dt=1, min_fades=30):
    fade_mask = I < threshold
    fade_start = numpy.where(numpy.diff(fade_mask.astype(int)) == 1)[0] + 1
    fades = numpy.array_split(fade_mask, fade_start)[1:]

    # select only fades that end within the window (i.e. final element is not fading)
    fades_filt = [i for i in fades if i[-1] != True]

    if len(fades_filt) < min_fades:
        # not enough fades to characterise duration, return nan
        return numpy.nan

    fade_durs = [i.sum() for i in fades_filt]
    mn = numpy.mean(fade_durs)
    return mn * dt


def ber_ook(EbN0, samples=None):
    '''
    Bit Error Rate for On-Off-Keying communications channel.

    Monte Carlo integration of Eq. 58 from Andrews and Phillips (2005) Ch 11.
    Note that electrical SNR per bit (Eb/N0) is used, not OSNR as in A+P.

    Args:
        EbN0 (float): average signal-to-noise ratio per bit Eb/N0 in electrical domain, dB
        samples (numpy.ndarray, optional): random samples of received power from FAST.
            If None is provided, assume no atmosphere (i.e. intensity pdf = delta function)

    Returns:
        Bit error rate for OOK
    '''
    snr = numpy.sqrt(10**(EbN0/10))

    if samples is None:
        # return BER assuming no atmospheric spreading
        return Q(snr)

    # Normalise samples by mean
    s = samples/samples.mean()

    return Q(s * snr).mean()


def sep_qam(M, EsN0, samples=None):
    '''
    Symbol error probabilty for square M-ary QAM, from Rice.

    Parameters:
        M (int): Number of symbols (must be perfect square)
        EsN0 (float): Average electrical symbol signal-to-noise ratio [dB]
        samples (numpy.ndarray, optional): random samples of received power from FAST.
    '''
    EsN0_frac = 10**(EsN0/10)
    prefactor = (numpy.sqrt(M)-1)/numpy.sqrt(M)

    if samples is None:
        return 4*(prefactor*Q(numpy.sqrt(3/(M-1) * EsN0_frac)) - prefactor**2 * Q(numpy.sqrt(3/(M-1) * EsN0_frac))**2)

    s = samples / samples.mean()
    EsN0_frac *= s**2

    return 4*(prefactor*Q(numpy.sqrt(3/(M-1) * EsN0_frac)) - prefactor**2 * Q(numpy.sqrt(3/(M-1) * EsN0_frac))**2).mean()


def ber_qam(M, EbN0, samples=None):
    '''
    Bit error rate for square M-ary QAM, from Rice. Assumes only nearest neighbour
    bit errors and Gray coding, i.e. 1 bit error per symbol error.

    Parameters:
        M (int): Number of symbols (must be perfect square)
        EbN0 (float): Average electrical signal-to-noise ratio per bit [dB]
        samples (numpy.ndarray, optional): random samples of received power from FAST.
    '''
    return 1/numpy.log2(M) * sep_qam(M, 10*numpy.log10(numpy.log2(M)) + EbN0, samples)


def Q(x):
    '''
    Q function from Rice book
    '''
    return 1/2 * erfc(x/numpy.sqrt(2))


def generalised_mutual_information_qam(samples, M, npxls, EsN0, N0=None, shot=False):
    '''
    Generalised Mutual Information (GMI), adapted from Alvarado et al (2016)
    [10.1109/JLT.2015.2450537] and Cho et al (2017) [10.1109/ECOC.2017.8345872].

    Assumes
        1) Perfect interleaving (no correlation between bits)
        2) Bit-wise decoder (no memory of other bits sent)
        3) Gray encoding of QAM symbols
        4) Soft decision decoding with FEC

    Args:
        samples (numpy.ndarray): Array of Monte Carlo complex field measurements or amplitudes
        M (int): number of symbols (must be perfect square)
        npxls (int): Number of pixels to use for binning
        EsN0 (float): Symbol signal-to-noise ratio [dB]
        N0 (float, optional): Noise variance (overrides EsN0, can be set to 0)

    Returns:
        GMI (float): the generalised mutual information value, in bits/symbol
    '''
    fyx = convolve_awgn_qam(samples, M, npxls, EsN0, N0=N0, region_size="full", shot=shot)
    fy = fyx.mean(0)
    log2_fy = numpy.ma.log2(fy)

    gray_code = _bin2gray_qam(M)
    m = int(numpy.log2(M))
    gmi = numpy.zeros((m, 2, npxls, npxls))
    for i in range(m):
        ix = _bit_at_index(gray_code, i, 0)
        fyb_0 = fyx[ix].mean(0)
        fyb_1 = fyx[~ix].mean(0)
        log2_fyb_0 = numpy.ma.log2(fyb_0)
        log2_fyb_1 = numpy.ma.log2(fyb_1)
        gmi[i,0] = fyb_0 * (log2_fyb_0 - log2_fy)
        gmi[i,1] = fyb_1 * (log2_fyb_1 - log2_fy)

    return gmi.sum((-1,-2)).mean(1).sum()


def mutual_information_qam(samples, M, npxls, EsN0, N0=None, shot=False):
    '''
    Equation 16 from Alvarado et al (2016) 10.1109/JLT.2015.2450537.

    This is for a memoryless receiver (no knowledge of other bits transmitted)
    '''
    fyx = convolve_awgn_qam(samples, M, npxls, EsN0, N0=N0, region_size="full", shot=shot)

    fy = fyx.mean(0)
    return (fyx * (numpy.ma.log2(fyx)-numpy.ma.log2(fy))).sum((-1,-2)).mean()


def convolve_awgn_qam(samples, M, npxls, EsN0, N0=None, region_size="individual", shot=False):
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
            logger.debug("AWGN noise level too large for region, increasing region size")
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

    x = numpy.linspace(-decision_region_size_norm/2,decision_region_size_norm/2,npxls+1)
    y = x.copy()

    for c in range(len(constellation)):
        xbin = x.copy()
        ybin = y.copy()

        if region_size == "individual":
            xbin += constellation_norm[c].real
            ybin += constellation_norm[c].imag

        samples_norm = constellation[c] * numpy.abs(samples)
        h = numpy.histogram2d(samples_norm.real, samples_norm.imag, bins=[xbin,ybin])[0] / len(samples_norm)

        if not shot:
            # perform separated 2d gaussian filter
            h_conv = correlate1d(h, g, mode='constant', axis=0)
            h_conv = correlate1d(h_conv, g, mode='constant', axis=1)
        else:
            ix, iy = numpy.where(h>0)
            sigma_mults = numpy.abs(samples).mean()**2 / (xbin[ix]**2 + ybin[iy]**2)
            h_conv = numpy.zeros(h.shape)
            for i in range(len(sigma_mults)):
                h_conv += \
                    h[ix[i],iy[i]] * \
                    gaussian2d(h.shape, numpy.sqrt(sigma2*sigma_mults[i]/2), cent=(ix[i], iy[i])) / (numpy.pi * sigma2 * sigma_mults[i])

        out[c] = h_conv

    return out


def define_constellation(modulation):
    '''
    Define constellations for coherent modulation schemes. Schemes supported:

    OOK - On-Off Keying
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
    if modulation == "OOK":
        nsymbols = 2
        constellation = numpy.array([0,1])

    # coherent schemes
    elif modulation == "BPSK":
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


def _bin2gray_qam(M):

    m = int(numpy.log2(M))
    symbols = numpy.arange(M, dtype=int)
    symbols_bin = [bin(i)[2:].zfill(m) for i in symbols]
    symbols_gray = []

    for s in symbols_bin:

        sgray = s[0]
        for i in range(len(s)-1):
            sgray += str(int(s[i]) ^ int(s[i+1]))

        symbols_gray.append(sgray)

    # flip every other row
    nside = int(numpy.sqrt(M))
    tmp = numpy.array(symbols_gray).reshape(nside, nside).copy()
    for i in tmp[1::2]:
        i[:] = i[::-1]

    symbols_gray = tmp.flatten()

    return symbols_gray


def _bit_at_index(code, index, bit):

    bit = str(bit)
    out = numpy.zeros(len(code), dtype=bool)
    for i,c in enumerate(code):
        out[i] = c[index] == str(bit)

    return out


def _encode(bs, bps):
    a = numpy.frombuffer(bs, dtype=numpy.uint8)
    bits = numpy.unpackbits(a)
    pad_bits = 0

    # if bps = 1 we are done
    if bps == 1:
        return bits, pad_bits

    r = len(bits)%bps
    if r > 0:
        pad_bits = bps -r
        bits = numpy.pad(bits, [0,pad_bits])
    symbols = (bits.reshape(-1, bps) * 2**(numpy.arange(bps, dtype=numpy.uint8)[::-1])).sum(1).flatten().astype(numpy.uint8)
    return symbols, pad_bits


def _decode(symbols, bps, pad_bits=0):

    # in the common case where bps = 1, symbols are the bits
    if bps == 1:
        return numpy.packbits(symbols)

    # unpack bits of symbol array, cut off the unused bits
    bits = numpy.unpackbits(symbols).reshape(-1,8)[:,-bps:].flatten()
    return numpy.packbits(bits).tobytes()[:-(pad_bits>0) or None]


def flip_bits(data, ber):
    if isinstance(data, str):
        b = data.encode("ascii")
    elif isinstance(data, numpy.ndarray):
        b = data.tobytes()
    else:
        raise Exception("String or numpy array as data please")

    bits = numpy.unpackbits(numpy.frombuffer(b, dtype=numpy.uint8))
    # flip_ix = numpy.random.choice(len(bits), size=int(len(bits) * ber), replace=False)
    flip_ix = numpy.where(numpy.random.rand(len(bits)) < ber)[0]
    bits[flip_ix] ^= 1

    newbytes = numpy.packbits(bits)
    if isinstance(data, str):
        newdata = (newbytes%128).tobytes().decode("ascii")
    else:
        newdata = numpy.frombuffer(newbytes.tobytes(), dtype=data.dtype).reshape(data.shape)

    return newdata
