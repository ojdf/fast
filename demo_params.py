'''
Test parameter file for FAST
'''
import numpy
from fast import turbulence_models
from aotools import equivalent_layers

h = numpy.arange(25000)
cn2 = turbulence_models.HV57(h)
h_el, cn2dh_el = equivalent_layers(h, cn2, 7)

w = turbulence_models.Bufton_wind(h_el)

p = {
'NPXLS': 'auto',                            # Number of sim pixels (can be "auto") 
'DX': 'auto',                               # Pixel scale (can be "auto") [m/pixel] 
'NITER': 100,                               # Number of random iterations
'SUBHARM': True,                            # Include subharmonics
'FFTW': False,                              # Use pyfftw
'FFTW_THREADS': 1,                          # Number of fftw threads
'NCHUNKS': 10,                              # Number of chunks to split Niter into (reduces memory requirements)
'TEMPORAL': False,                          # Generate temporal irradiance sequences
'DT': 0.001,                                # Simulation timestep (if temporal sequences used)

'W0': (1.016-0.35)/2/numpy.sqrt(8),         # 1/e^2 launch beam radius [m]
'F0': numpy.inf,                            # Launch beam radius of curvature [m]
'Tx': 1.016,                                # Diameter of circular ground aperture [m]
'Tx_obsc': 0.35,                            # Diameter of central obscuration of ground aperture [m]
'Rx': 0.01,                                 # Diameter of circular reciever aperture [m]
'DTHETA': [4,0],                            # Point ahead (x,y) [arcseconds]
'WVL': 1064e-9,                             # Laser wavelength [m]
'AXICON': True,                             # Axicon (donut) launch shape
'POWER': 20,                                # Laser power [W]
'SMF': False,                               # Use single mode fibre (downlink only)

'H_SAT': 36e6,                              # Satellite height above ground [m]
'H_TURB': h_el,                             # Turbulence altitudes [m]
'CN2_TURB': cn2dh_el,                       # Cn2(dh) per turbulence layer [m^-2/3 or m^1/3]
'WIND_SPD': w,                              # Wind speed per layer
'WIND_DIR': numpy.ones(len(h_el)) * 90,     # Wind direction per layer
'L0': numpy.inf,                            # Turbulence outer scale [m]
'l0': 1e-6,                                 # Turbulence inner scale [m]
'C': 2*numpy.pi,                            # Turbulence power spectrum constant
'LAYER': True,                              # Cn2 or Cn2dh values (True = Cn2dh)
'ZENITH_ANGLE': 55,                         # Zenith angle [degrees]
'PROP_DIR': 'up',                           # Uplink ('up') or downlink ('down') propagation

'AO_MODE': 'AO_PA',                         # AO mode ('AO', 'AO_PA', 'TT_PA', 'LGS_PA', 'NOAO')
'DSUBAP': 0.02,                             # WFS subaperture pitch [m]
'TLOOP': 0.001,                             # AO loop delay [s]
'TEXP': 0.001,                              # WFS exposure time
'ALIAS': True,                              # Include WFS aliasing
'NOISE': 1.,                                # WFS noise [rad^2]
'MODAL': False,                             # Modal (True) or Zonal (False) correction
'MODAL_MULT': 1,                            # Multiplier to reduce number of modes if required
'ZMAX': None,
'GTILT': False                             # Use G tilt or Z tilt for Tip/Tilt calcs

}