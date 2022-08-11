'''
Functions regarding optical communications
'''
import numpy


class Modulator():

    def __init__(self, signal, scheme=None):
        self.signal = signal
        self.scheme = scheme

    def generate_symbols(self):

        if self.scheme in ['OOK', 'BPSK']:
            self.nsymbols = 2

        elif self.scheme in ["QPSK", "QAM"]:
            self.nsymbols = 4
            
        elif len(self.scheme.split("-")) ==2:
            self.nsymbols = int(self.scheme.split("-")[0])

        else:
            raise ValueError("Scheme not recognised")

        self.symbols = numpy.random.randint(0, self.nsymbols, size=len(self.signal))

    def modulate(self):

        if self.scheme == None:
            self.recv_signal = self.signal
            return self.recv_signal

        self.generate_symbols()
    
        # incoherent on-off keying
        if self.scheme == "OOK":
            if self.signal.dtype == complex:
                self.signal = numpy.abs(self.signal)**2

            self.recv_signal = self.symbols * self.signal
            return self.recv_signal

        # coherent schemes
        if self.signal.dtype != complex:
            raise ValueError(f"{self.scheme} modulation requires COHERENT=True!")

        elif self.scheme == "BPSK":
            # binary phase shift keying
            mod = numpy.exp(1j * self.symbols * numpy.pi)
        
        elif self.scheme in ["QPSK", "QAM"]:
            # quadrature PSK, quadrature amplitude modulation (same thing?)
            mod = numpy.exp(1j * ((self.symbols * numpy.pi/2) - numpy.pi/4))

        elif (self.scheme[-4:] == "-PSK"):
            # N-PSK
            mod = numpy.exp(1j * (self.symbols * numpy.pi/(self.nsymbols/2)))

        elif (self.scheme[-4:] == "-QAM"):
            # N-QAM
            n_side = int(numpy.sqrt(self.nsymbols))
            x = numpy.linspace(-1,1,n_side)
            xx, yy = numpy.meshgrid(x,x)
            pos = (xx + 1j * yy).flatten()

            mod = pos[self.symbols]

        else:
            raise ValueError(f"Modulation not found for scheme {self.scheme}")

        self.recv_signal = mod * self.signal
        return self.recv_signal

    def demodulate(self):
        raise NotImplementedError()

    


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