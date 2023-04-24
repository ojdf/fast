import pytest
import fast 
import numpy

p = None
def run_sim(p):
    sim = fast.Fast(p)
    sim.run()
    assert numpy.isfinite(sim.I).all()

# Test turbulence models
def test_HV57():
    h = numpy.linspace(0,20000,10)
    cn2 = fast.turbulence_models.HV57(h)
    assert len(cn2) == len(h)
    assert cn2.dtype == float

def test_Bufton():
    h = numpy.linspace(0,20000,10)
    w = fast.turbulence_models.Bufton_wind(h)
    assert len(w) == len(h)
    assert w.dtype == float

def test_HV57_Bufton():
    N = 10
    h, cn2, w = fast.turbulence_models.HV57_Bufton_profile(N)
    assert len(h) == len(cn2) == len(w)


# Test config parsing 
def test_config_default():
    c = fast.conf.ConfigParser("test/test_params.py")


# Test complete simulation
def test_sim_default():
    sim = fast.Fast("test/test_params.py")
    sim.run()
    global p
    p = sim.params.copy()
    assert numpy.isfinite(sim.result.power).all()
    assert numpy.isfinite(sim.result.dB_rel).all()
    assert numpy.isfinite(sim.result.dB_abs).all()

@pytest.mark.skipif(not fast.fast._pyfftw, reason="No pyfftw installed")
def test_sim_fftw():
    ptmp = p.copy()
    ptmp['FFTW'] = True
    run_sim(ptmp)

def test_sim_temporal():
    ptmp = p.copy()
    ptmp['TEMPORAL'] = True
    run_sim(ptmp)

def test_sim_obsc():
    ptmp = p.copy()
    ptmp['OBSC_GROUND'] = 0.1
    run_sim(ptmp)

def test_sim_obsc_sat():
    ptmp = p.copy()
    ptmp['OBSC_SAT'] = 0.05
    run_sim(ptmp)

def test_sim_axicon():
    ptmp = p.copy()
    ptmp['AXICON'] = True
    ptmp['OBSC_GROUND'] = 0.1
    run_sim(ptmp)

def test_sim_L_SAT():
    ptmp = p.copy()
    ptmp['L_SAT'] = 500e3
    sim = fast.Fast(ptmp)
    assert sim.L == ptmp['L_SAT']

def test_sim_L0():
    ptmp = p.copy()
    ptmp['L0'] = 25
    run_sim(ptmp)

def test_sim_down():
    ptmp = p.copy()
    p['PROP_DIR'] = 'down'
    run_sim(p)

def test_sim_NOAO():
    ptmp = p.copy()
    ptmp['AO_MODE'] = 'NOAO'
    run_sim(ptmp)

def test_sim_TT():
    ptmp = p.copy()
    ptmp['AO_MODE'] = 'TT'
    run_sim(ptmp)

def test_sim_noise():
    ptmp = p.copy()
    p['NOISE'] = 1
    run_sim(ptmp)

def test_sim_modal():
    ptmp = p.copy()
    ptmp['MODAL'] = True
    run_sim(ptmp)


# Test comms 
def test_sim_coherent():
    ptmp = p.copy()
    ptmp['COHERENT'] = True
    sim = fast.Fast(ptmp)
    sim.run()
    assert sim.I.dtype == complex

