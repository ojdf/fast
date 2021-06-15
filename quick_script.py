import fsocffs
from demo_params import p

p['NITER'] = 10000
p['NCHUNKS'] = 10
p['DT'] = 0.0003
p['AO_MODE'] = 'NOAO'
p['TEMPORAL'] = True
p['ALIAS'] = False
p['NOISE'] = 0
p['L0'] = numpy.inf
p['SUBHARM'] = True

p['H_TURB'] = numpy.array([0])
p['CN2_TURB'] = numpy.array([100e-15])
p['WIND_SPD'] = numpy.array([10])
p['WIND_DIR'] = numpy.array([0])

sim = fsocffs.FFS(p)

# r1 = fsocffs.funcs.generate_random_coefficients(sim.Niter//sim.Nchunks, sim.powerspec, sim.temporal, sim.temporal_powerspec)
# r2 = fsocffs.funcs.generate_random_coefficients(sim.Niter//sim.Nchunks, sim.powerspec, False, sim.temporal_powerspec)
sim.run()