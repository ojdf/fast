'''
Test parameter file for FAST
'''
import numpy
from fast import turbulence_models

# define turbulence profile and wind speeds
h = numpy.array([0,5000,10000,15000])
cn2 = turbulence_models.HV57(h) * 100
w = turbulence_models.Bufton_wind(h)
wdir = [0,90,180,270]

p = {
'NPXLS': 'auto',                            # Number of sim pixels (can be "auto") 
'DX': 'auto',                               # Pixel scale (can be "auto") [m/pixel] 
'NITER': 1000,                               # Number of random iterations
'SUBHARM': True,                            # Include subharmonics
'FFTW': False,                              # Use pyfftw
'FFTW_THREADS': 1,                          # Number of fftw threads
'NCHUNKS': 10,                              # Number of chunks to split Niter into (reduces memory requirements)
'TEMPORAL': False,                          # Generate temporal irradiance sequences
'DT': 0.001,                                # Simulation timestep (if temporal sequences used)

'W0': 0.3,                                  # 1/e^2 launch beam radius [m]
'F0': numpy.inf,                            # Launch beam radius of curvature [m]
'Tx': 0.8,                                  # Diameter of circular ground aperture [m]
'Tx_obsc': 0,                               # Diameter of central obscuration of ground aperture [m]
'Rx': 0.01,                                 # Diameter of circular reciever aperture [m]
'DTHETA': [4,0],                            # Point ahead (x,y) [arcseconds]
'WVL': 1550e-9,                             # Laser wavelength [m]
'AXICON': False,                            # Axicon (donut) launch shape
'POWER': 20,                                # Laser power [W]
'SMF': False,                               # Use single mode fibre (downlink only)
'COHERENT': False,                          # Coherent detection (SMF only)
'MODULATION': "OOK",

'H_SAT': 36e6,                              # Satellite height above ground [m]
'H_TURB': h,                                # Turbulence altitudes [m]
'CN2_TURB': cn2,                            # Cn2(dh) per turbulence layer [m^-2/3 or m^1/3]
'WIND_SPD': w,                              # Wind speed per layer [m/s]
'WIND_DIR': wdir,                           # Wind direction per layer [degrees]
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
'NOISE': 0,                                # WFS noise [rad^2 ?]
'MODAL': False,                             # Modal (True) or Zonal (False) correction
'MODAL_MULT': 1,                            # Multiplier to reduce number of modes if required
'ZMAX': None,
'GTILT': False                              # Use G tilt or Z tilt for Tip/Tilt calcs

}