import fast
from demo_params import p
import numpy

p['NITER'] = 1000
p['NCHUNKS'] = 10
p['DT'] = 0.001
p['AO_MODE'] = 'AO_PA'
p['TEMPORAL'] = True
p['ALIAS'] = False 
p['NOISE'] = 0
p['L0'] = 25
p['SUBHARM'] = True
p['TLOOP'] = 0.001
p['TEXP'] = 0
p['DSUBAP'] = 1/8.
p['DX'] = 0.05
p['NPXLS'] = 128
p['DTHETA'] = [0,0]
p['ZENITH_ANGLE'] = 0

p['H_TURB'] = numpy.array([0,5000,10000])
p['CN2_TURB'] = numpy.array([0.5,0.3,0.1]) * 100e-15
p['WIND_SPD'] = numpy.array([10,20,30])
p['WIND_DIR'] = numpy.array([0,45,90])

sim = fast.Fast(p)

sim.run()