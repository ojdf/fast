import numpy
from aotools.turbulence.profile_compression import equivalent_layers

def HV57(h, w=21, A=1.7e-14):
    '''
    Hufnagel Valley 5/7 Cn2 profile. 

    NOTE this returns Cn2 [m^-2/3], not integrated Cn2dh [m^1/3] per layer.

    Params:
        h (numpy.ndarray): height (m)
        w (float): "wind" parameter
        A (float): "A" parameter

    Returns:
        cn2 (numpy.ndarray): cn2 per layer
   '''
    cn2 = 0.00594 * (w/27)**2 * (1e-5 * h)**10 * numpy.exp(-h/1000) + 2.7e-16 * numpy.exp(-h/1500) + A * numpy.exp(-h/100.)
    return cn2


def Bufton_wind(h, vg=8, vt=30, ht=9400., Lt=4800.):
    '''
    Bufton wind speed model. 

    See e.g. Roberts & Bradford, Optics Express. 19(2):820-37 [2011].

    Parameters:
        h (numpy.ndarray): height (m)
        vg (float): ground/low altitude wind speed (m/s)
        vt (float): tropopause wind speed (m/s)
        ht (float): height of tropopause
        Lt (float): thickness of tropopause

    Returns:
        wind_speed (numpy.ndarray): wind speed per layer (m/s)
    '''
    return vg + vt * numpy.exp(-((h-ht)/Lt)**2)


def HV57_Bufton_profile(N, w=21, A=1.7e-14, vg=8, vt=30, ht=9400., Lt=4800.):
    '''
    Produce a turbulence and wind profile of N layers using HV57 for Cn2 and 
    Bufton for wind speed. First creates a high-resolution Cn2 profile then 
    uses equivalent layers (see aotools.profile_compression.equivalent_layers) 
    to reduce to N layers. 

    Params:
        N (int): number of layers
        kwargs: keyword arguments for non-default HV57 and Bufton models

    Returns:
        h (numpy.ndarray): height of layers (m)
        cn2dh (numpy.ndarray): integrated Cn2dh per layer (m^1/3)
        wind (numpy.ndarray): wind speed per layer (m/s)
    '''
    h0 = numpy.arange(0,30000) # 1m bins up to 30km
    cn20 = HV57(h0, w, A)
    w0 = Bufton_wind(h0, vg, vt, ht, Lt)

    h, cn2, w = equivalent_layers(h0, cn20, N, w=w0)
    return h, cn2, w


def equivalent_layers(h, p, L, w=None):
    '''
    STOLEN FROM AOTOOLS (https://github.com/AOtools/aotools/blob/main/aotools/turbulence/profile_compression.py)
    SINCE THIS FUNCTION IS NOT CURRENTLY IN LATEST AOTOOLS RELEASE 1.0.7

    Equivalent layers method of profile compression (Fusco 1999).
    Splits the profile into L "slabs", then sets the height of each slab as the 
    effective height ((integral cn2(h) * h^{5/3} dh) / integral cn2(h) dh)^(3/5)
    and the cn2 as the sum of cn2 in that slab.
    Can also provide wind speed per layer, in which case the wind speeds are calculated 
    per layer in a similar fashion ((integral cn2(h) * w^{5/3} dh) / integral cn2(h) dh)^(3/5)
    for wind speed w. This conserves coherence time as well as isoplanatic angle.
    Parameters
        h (numpy.ndarray): heights of input profile layers
        p (numpy.ndarray): cn2dh values of input profile layers
        L (int): number of layers to compress down to
        w (numpy.ndarray, optional): wind speeds of input profile layers
    
    Returns
        h_L (numpy.ndarray): compressed profile heights
        cn2_L (numpy.ndarray): compressed profile cn2dh per layer 
        w (numpy.ndarray, optional): compressed profile wind speed per layer
    '''
    h_el = numpy.zeros(L)
    cn2_el = numpy.zeros(L)
    if w is not None:
        w_el = numpy.zeros(L)

    hstep = (h.max()-h.min())/L
    alt_bins = numpy.arange(h.min(), h.max(), hstep)
    ix = numpy.digitize(h, alt_bins)
    for i in range(L):
        ix_tmp = ix==i+1
        cn2_el[i] = p[ix_tmp].sum()
        h_el[i] = ((p[ix_tmp] * h[ix_tmp]**(5/3)).sum() / p[ix_tmp].sum())**(3/5)
        if w is not None:
            w_el[i] = ((p[ix_tmp] * w[ix_tmp]**(5/3)).sum() / p[ix_tmp].sum())**(3/5)

    if w is not None:
        return h_el, cn2_el, w_el

    return h_el, cn2_el