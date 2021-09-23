import numpy
from scipy.special import j1, jv, iv, jve
from . import funcs 
from aotools.functions.zernike import zernIndex
from aotools import cn2_to_r0, fouriertransform
from scipy.interpolate import RectBivariateSpline
from . import fsocffs

def zernike_ft(fabs, phi, D, n_noll):
    n, m = zernIndex(n_noll)
    if m == 0:
        return numpy.sqrt(n+1) * (-1)**(n/2.) * (2 * jv(n+1, fabs * D / 2) / (fabs * D/2))
    elif n_noll % 2 == 0:
        return numpy.sqrt(2 * (n+1)) * (-1)**((n-m)/2.) * (1j)**m * (2 * jv(n+1, fabs * D / 2) / (fabs * D/2)) * numpy.cos(m * phi)
    else:
        return numpy.sqrt(2 * (n+1)) * (-1)**((n-m)/2.) * (1j)**m * (2 * jv(n+1, fabs * D / 2) / (fabs * D/2)) * numpy.sin(m * phi)
    
def zernike_filter(fabs, fx, fy, D, n_noll, n_noll_start=1, gamma=None):
    phi = numpy.arctan2(fy,fx)

    if gamma is None:
        out = numpy.zeros(fabs.shape, dtype=complex)

        for i in range(n_noll_start, n_noll+1):
            out += zernike_ft(fabs, phi, D, i)
        mid = int(fabs.shape[0]/2)

        if n_noll_start == 1:
            out[mid, mid] = 1
        else:
            out[mid,mid] = 0
        return out

    else:
        if numpy.array(gamma).ndim == 0:
            gamma = numpy.array([gamma])
        out = numpy.zeros((len(gamma), fabs.shape[0], fabs.shape[1]), dtype=complex)
        for ix,g in enumerate(gamma):
            for i in range(n_noll_start, n_noll+1):
                out[ix] += zernike_ft(fabs, phi, g*D, i)

        mid = int(fabs.shape[0]/2)
        if n_noll_start == 1:
            out[:,mid, mid] = 1
        else:
            out[:,mid,mid] = 0
        return out

def zernike_squared_filter(fabs, fx, fy, D, n_noll, n_noll_start=1, gamma=None, plusminus=False):
    phi = numpy.arctan2(fy,fx)
    if plusminus:
        phi1 = numpy.arctan2(-fy,-fx)

    if gamma is None:
        out = numpy.zeros(fabs.shape, dtype=complex)

        for i in range(n_noll_start, n_noll+1):
            z = zernike_ft(fabs, phi, D, i)
            if plusminus:
                z1 = zernike_ft(fabs, phi1, D, i)
                out += z * numpy.conjugate(z1)
            else:
                out += numpy.abs(z)**2
        mid = int(fabs.shape[-1]/2)

        if n_noll_start == 1:
            out[...,mid, mid] = 1
        else:
            out[...,mid,mid] = 0
        return out

    else:
        if numpy.array(gamma).ndim == 0:
            gamma = numpy.array([gamma])
        out = numpy.zeros((len(gamma), fabs.shape[0], fabs.shape[1]), dtype=complex)
        for ix,g in enumerate(gamma):
            for i in range(n_noll_start, n_noll+1):
                z = zernike_ft(fabs, phi, g*D, i)
                if plusminus:
                    z1 = zernike_ft(fabs, phi1, g*D, i)
                    out[ix] += z * numpy.conjugate(z1)
                else:
                    out[ix] += numpy.abs(z)**2
        mid = int(fabs.shape[0]/2)
        if n_noll_start == 1:
            out[:,mid, mid] = 1
        else:
            out[:,mid,mid] = 0
        return out

def piston_gtilt_filter(fabs, fx, fy, D):
    pist = zernike_squared_filter(fabs, fx, fy, D, 1)
    G_tt = jv(1, fabs * D/2.)**2
    filt = (pist + G_tt).real
    filt[filt>1] = 1.
    return filt

def piston_filter(fabs, D):
    filt = 1 - (2*j1(0.5 * D * fabs) / (0.5 * D * fabs))**2
    filt[int(fabs.shape[0]/2), int(fabs.shape[1]/2)] = 0
    return filt

def tiptilt_filter(fabs, D):
    filt = 1 - (4 * jv(2, 0.5 * D * fabs) / (0.5 * D * fabs))**2
    filt[int(fabs.shape[0]/2), int(fabs.shape[1]/2)] = 1
    return filt

def piston_tiptilt_filter(fabs, D):
    filt = 1 - (2*j1(0.5 * D * fabs) / (0.5 * D * fabs))**2 - (4 * jv(2, 0.5 * D * fabs) / (0.5 * D * fabs))**2
    filt[int(fabs.shape[0]/2), int(fabs.shape[1]/2)] = 0
    return filt

def mask_lf(freq, d_WFS, modal=False, modal_mult=1, Zmax=None, D=None, Gtilt=False):
    fx = freq.fx
    fy = freq.fy

    fmax = numpy.pi/d_WFS
    wfs_space = numpy.logical_and(abs(fx) <= fmax, abs(fy) <= fmax)
    if modal:
        fabs = numpy.sqrt(fx**2 + fy**2)
        if Zmax is None:
            dm_space = fabs <= fmax * modal_mult
        else:
            if Gtilt:
                dm_space = piston_gtilt_filter(fabs, fx, fy, D)
            else:
                dm_space = zernike_squared_filter(fabs, fx, fy, D, Zmax).real
    else:
        dm_space = wfs_space

    mask = wfs_space * dm_space
    return mask

def mask_hf(freq, d_WFS, modal=False, modal_mult=1, Zmax=None, D=None, Gtilt=False):
    fx = freq.fx
    fy = freq.fy
    return 1 - mask_lf(fx, fy, d_WFS, modal=modal, modal_mult=modal_mult, Zmax=Zmax, D=D, Gtilt=Gtilt)

def Jol_noise_openloop(freq, Dsubap, noise_variance, lf_mask):
    fabs = freq.fabs
    fx = freq.fx
    fy = freq.fy
    
    N = noise_variance #* (Dsubap/(2*numpy.pi))**2
    if freq.freq_per_layer:
        N /= fabs.shape[0]
    powerspec = N / (fabs**2 * numpy.sinc(Dsubap * fx / (2*numpy.pi))**2 * numpy.sinc(Dsubap * fy / (2*numpy.pi))**2)
    midpt_x = int(powerspec.shape[-2]/2.)
    midpt_y = int(powerspec.shape[-1]/2.)
    powerspec[...,midpt_x, midpt_y] = 0.

    return lf_mask * powerspec

def Jol_alias_openloop(freq, Dsubap, p, lf_mask, v=None, Delta_t=None, wvl=None, lmax=10, kmax=10, L0=numpy.inf, l0=1e-6):
    fx = freq.fx
    fy = freq.fy
    fabs = freq.fabs

    ls = numpy.arange(-lmax, lmax+1)
    ks = numpy.arange(-kmax, kmax+1)
    midpt_x = int(fx.shape[-2]/2.)
    midpt_y = int(fy.shape[-1]/2.)

    if freq.freq_per_layer:
        fx_tile = fx
        fy_tile = fy
        alias = numpy.zeros(fabs.shape)
    else:
        fx_tile = numpy.tile(fx, (len(p),*[1]*fx.ndim))
        fy_tile = numpy.tile(fy, (len(p),*[1]*fy.ndim))
        alias = numpy.zeros((len(p), *fabs.shape))

    if v is not None:
        v_dot_kappa = (fx_tile.T * v[:,0] + fy_tile.T * v[:,1]).T
    else:
        v_dot_kappa = 0

    sinc_term = numpy.sinc(Delta_t * v_dot_kappa / (2*numpy.pi))**2

    #TODO: the central pixel may still be wrong
    for l in ls:
        for k in ks:
            if l == 0 and k == 0:
                continue

            # Quick and dirty SpatialFrequencyStruct here, probably not optimal
            fx_shift = freq.fx_axis - 2*numpy.pi * k/Dsubap
            fy_shift = freq.fy_axis - 2*numpy.pi * l/Dsubap
            freq_shift = fsocffs.SpatialFrequencyStruct(fx_shift, fy_shift, freq_per_layer=freq.freq_per_layer)

            term_1 = (freq.fx/(freq_shift.fy) + freq.fy/(freq_shift.fx))**2
            term_2 = funcs.turb_powerspectrum_vonKarman(freq_shift, p, L0=L0, l0=l0)
            mult = term_1 * term_2 *  freq.fx**2 * freq.fy**2 / freq.fabs**4
            mult[...,midpt_x,midpt_y] = 0.
            if l == 0:
                mult[...,midpt_x,:] = term_2[...,midpt_x,:]
            if k == 0:
                mult[...,midpt_y] = term_2[...,midpt_y]
                mult[...,midpt_x,midpt_y] = term_2[...,midpt_x,midpt_y]
            alias += mult

    alias *= sinc_term * lf_mask

    # there is a small chance that there are still some NaNs or infs in the 
    # array, for specific combinations of freqency axes and subapertures.
    # Hopefully these lie outside the mask area so just set them to 0.
    alias[numpy.isnan(alias)] = 0.

    return alias

def G_AO_Jol(freq, mask, mode='AO', h=None, v=None,  dtheta=[0,0], Tx=None, 
            wvl=None, Zmax=None, tl=0, Delta_t=0, Dsubap=None, modal=False, modal_mult=1):

    fabs = freq.fabs
    fx = freq.fx
    fy = freq.fy

    if mode not in ['NOAO', 'AO', 'AO_PA', 'TT_PA', 'LGS_PA']:
        raise Exception('Mode not recognised')

    if mode is 'NOAO':
        return 1 

    if mode is 'AO':
        return 1-mask

    if freq.freq_per_layer:
        fx_tile = fx
        fy_tile = fy
    else:
        fx_tile = numpy.tile(fx, (len(h),*[1]*fx.ndim))
        fy_tile = numpy.tile(fy, (len(h),*[1]*fy.ndim))

    delta_r_theta = (numpy.tile(dtheta, (len(h),1)).T / 206265. * h ).T

    delta_r_dot_kappa = (fx_tile.T * delta_r_theta[:,0] + fy_tile.T * delta_r_theta[:,1]).T

    if v is not None:
        v_dot_kappa = (fx_tile.T * v[:,0] + fy_tile.T * v[:,1]).T
    else:
        v_dot_kappa = 0

    term_1 = 2 * numpy.cos(delta_r_dot_kappa - tl * v_dot_kappa)
    term_2 = numpy.sinc(Delta_t * v_dot_kappa / (2*numpy.pi))

    aniso = 1 - term_1 * term_2 + term_2**2

    if mode is 'AO_PA' or mode is 'TT_PA':
        return aniso * mask + (1-mask)

    if mode is 'LGS_PA':
        term_1_lgs = 2 * numpy.cos(-tl * v_dot_kappa)
        term_2_lgs = numpy.sinc(Delta_t * v_dot_kappa / (2*numpy.pi))
        aniso_lgs = 1 - term_1_lgs * term_2_lgs + term_2_lgs**2
        Z = zernike_squared_filter(fabs, fx, fy, Tx, 4, n_noll_start=1).real
        return mask * (Z * aniso + (1-Z) * aniso_lgs) + (1-mask)

    raise Exception("Shouldn't be here")

def logamp_powerspec(freq, h, cn2, wvl, pupilfilter=None, layer=True, L0=numpy.inf, l0=1e-6):

    fabs = freq.fabs

    if freq.freq_per_layer:
        fabs_3d = fabs
    else:
        fabs_3d = numpy.tile(fabs, (len(h),*[1]*fabs.ndim))

    powerspec = funcs.turb_powerspectrum_vonKarman(freq, cn2, L0=L0, l0=l0) * 2*numpy.pi*(2*numpy.pi/wvl)**2

    powerspec *= numpy.sin(wvl * h * fabs_3d.T**2 / (4 * numpy.pi)).T**2

    if pupilfilter is not None:
        if type(pupilfilter) is numpy.ndarray:
            powerspec *= pupilfilter
        elif type(pupilfilter) is RectBivariateSpline:
            # sample at correct spatial frequencies
            if freq.freq_per_layer:
                P = numpy.zeros(freq.fx.shape)
                for i in range(freq.fx_axis.shape[0]):
                    P[i] = pupilfilter(freq.fy_axis[i], freq.fx_axis[i])
            else:
                P = pupilfilter(freq.fy_axis, freq.fx_axis)
              
            powerspec *= P

    powerspec_integrated = funcs.integrate_path(powerspec, h, layer=layer)

    return powerspec_integrated

def DM_transfer_function(fx, fy, fabs, mode, Zmax=None, D=None, dsubap=None):
    if mode is 'perfect':
        return 1.

    elif mode is 'zernike':
        return zernike_filter(fabs, fx, fy, D, Zmax)

    else:
        raise NotImplementedError('Choose DM that is implemented')
        # TODO add more functions 

def G_AO_Jol_closedloop(fx, fy, fabs, h, dtheta=[0,0], Delta_t=0., tl=0., gloop=1., v=None, 
                        dsubap=None, DM='perfect', Zmax=None, D=None, nu=1, modal=False, modal_mult=1):

    Gamma_DM = DM_transfer_function(fx, fy, fabs, mode=DM, Zmax=Zmax, D=D, dsubap=dsubap)
    
    # convert to linear spatial frequencies because I can't be bothered to convert
    # the long expressions below 
    fx = fx.copy()/(2*numpy.pi)
    fy = fy.copy()/(2*numpy.pi)
    fabs = fabs.copy()/(2*numpy.pi)

    fx_tile = numpy.tile(fx, (len(h),1,1))
    fy_tile = numpy.tile(fy, (len(h),1,1))

    delta_r_theta = (numpy.tile(dtheta, (len(h),1)).T / 206265. * h ).T
    delta_r_dot_f = (fx_tile.T * delta_r_theta[:,0] + fy_tile.T * delta_r_theta[:,1]).T

    if v is not None:
        v_dot_f = (fx_tile.T * v[:,0] + fy_tile.T * v[:,1]).T
    else:
        v_dot_f = 0

    # aniso-servo
    F_AS_top = (1 + gloop**2 * Gamma_DM**2 * numpy.sinc(Delta_t * v_dot_f)**2 * (1 + nu**2 * Gamma_DM**2)/2.
                - numpy.cos(2*numpy.pi*Delta_t*v_dot_f) 
                
                + gloop * Gamma_DM**2 * numpy.sinc(Delta_t * v_dot_f) * nu *
                (numpy.cos(2*numpy.pi * delta_r_dot_f + 2*numpy.pi * (Delta_t/2 - tl) * v_dot_f) - 
                numpy.cos(2*numpy.pi * delta_r_dot_f - 2*numpy.pi * (Delta_t/2 + tl) * v_dot_f))
                
                + gloop * Gamma_DM * numpy.sinc(Delta_t * v_dot_f) * (numpy.cos(2*numpy.pi*(Delta_t/2 + tl) * v_dot_f) - 
                numpy.cos(2*numpy.pi * (Delta_t/2. - tl) * v_dot_f)) 
                
                - gloop**2 * Gamma_DM**3 * numpy.sinc(Delta_t * v_dot_f)**2 * nu * numpy.cos(2 * numpy.pi * delta_r_dot_f))

    F_AS_bottom = (1 + gloop**2 * Gamma_DM**2 * numpy.sinc(Delta_t * v_dot_f)**2/2. 
                    
                    + gloop * Gamma_DM * numpy.sinc(Delta_t * v_dot_f) * 
                    (numpy.cos(2*numpy.pi * (Delta_t/2. + tl) * v_dot_f) - numpy.cos(2*numpy.pi * (Delta_t/2. - tl) * v_dot_f))
                    
                    - numpy.cos(2*numpy.pi * Delta_t * v_dot_f))

    F_AS = F_AS_top/F_AS_bottom

    return F_AS