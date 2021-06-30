import fsocffs
from demo_params import p
import numpy

p['NITER'] = 1000
p['NCHUNKS'] = 10
p['DT'] = 0.001
p['AO_MODE'] = 'LGS_PA'
p['TEMPORAL'] = True
p['ALIAS'] = True 
p['NOISE'] = 1
p['L0'] = numpy.inf
p['SUBHARM'] = True

p['H_TURB'] = numpy.array([0,5000,1000])
p['CN2_TURB'] = numpy.array([0.5,0.3,0.1]) * 100e-15
p['WIND_SPD'] = numpy.array([10,20,30])
p['WIND_DIR'] = numpy.array([0,45,90])


sim = fsocffs.FFS(p)

# sim.run()