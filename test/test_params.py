'''
Test parameter file for FAST
'''
import numpy
from fast import turbulence_models

# define turbulence profile and wind speeds, directions
h, cn2, w = turbulence_models.HV57_Bufton_profile(4)
wdir = [0,90,180,270]

p = {
# Simulation parameters
'NPXLS': 'auto',                            # Number of sim pixels (can be "auto") 
'DX': 'auto',                               # Pixel scale (can be "auto") [m/pixel] 
'NITER': 1000,                              # Number of random iterations
'SUBHARM': True,                            # Include subharmonics
'FFTW': False,                              # Use pyfftw
'FFTW_THREADS': 1,                          # Number of fftw threads
'NCHUNKS': 10,                              # Number of chunks to split Niter into (reduces memory requirements)
'TEMPORAL': False,                          # Generate temporal irradiance sequences
'DT': 0.001,                                # Simulation timestep (if temporal sequences used)

# Transmitter/Receiver parameters
'WVL': 1550e-9,                             # Communications laser wavelength [m]
'POWER': 1,                                 # Transmitted laser power [W]
'W0': 0.3,                                  # 1/e^2 Tx beam radius (satellite/ground depending on PROP_DIR) [m]
'D_GROUND': 0.8,                            # Diameter of circular ground station aperture [m]
'OBSC_GROUND': 0,                           # Diameter of central obscuration of ground aperture [m]
'D_SAT': 0.1,                               # Diameter of circular satellite aperture [m] [!NOT USED!]
'OBSC_SAT': 0,                              # Diameter of central obscuration of satellite aperture
'AXICON': False,                            # Axicon (donut) launch shape
'SMF': True,                                # Include losses from coupling into single mode fibre at receiver

# Turbulence and Link parameters
'L_SAT': 36e6,                              # Distance from ground to satellite [m]
'H_TURB': h,                                # Turbulence altitudes [m]
'CN2_TURB': cn2,                            # Cn2dh per turbulence layer [m^1/3]
'WIND_SPD': w,                              # Wind speed per layer [m/s]
'WIND_DIR': wdir,                           # Wind direction per layer [degrees]
'L0': numpy.inf,                            # Turbulence outer scale [m]
'l0': 1e-6,                                 # Turbulence inner scale [m]
'ZENITH_ANGLE': 55,                         # Zenith angle [degrees]
'PROP_DIR': 'up',                           # Uplink ('up') or downlink ('down') propagation
'DTHETA': [4,0],                            # Point ahead (x,y) [arcseconds]
'TRANSMISSION': 1,                          # Atmospheric transmission coefficient

# Adaptive Optics parameters
'AO_MODE': 'AO',                            # AO mode (full AO 'AO', tip-tilt only 'TT', lgs-AO 'LGSAO', no AO 'NOAO')
'DSUBAP': 0.1,                              # WFS subaperture pitch [m]
'TLOOP': 0.001,                             # AO loop delay [s]
'TEXP': 0.001,                              # WFS exposure time
'ALIAS': True,                              # Include WFS aliasing
'NOISE': 0,                                 # WFS noise [rad^2 ?]
'MODAL': False,                             # Modal (True) or Zonal (False) correction
'MODAL_MULT': 1,                            # Multiplier to reduce number of modes if required
'ZMAX': None,                               # Maximum Zernike index for correction if modal [!NOT USED!]

# Communications parameters
'COHERENT': False,                          # Coherent detection 
'MODULATION': None,                         # Comms modulation scheme
'N0': 0,                                    # Detector noise [W], for AWGN

}