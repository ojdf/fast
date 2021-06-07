import fsocffs
from demo_params import p

p['NITER'] = 1000
p['AO_MODE'] = 'NOAO'
p['TEMPORAL'] = True
p['ALIAS'] = False
p['NOISE'] = 0
p['L0'] = 25
p['WIND_SPD'] = numpy.ones(7) * numpy.random.uniform(0,360,size=(7))

sim = fsocffs.FFS(p)

r = fsocffs.funcs.generate_random_coefficients(sim.Niter, sim.powerspec, sim.h, True, sim.temporal, sim.temporal_powerspec_integrated)