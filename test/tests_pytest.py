import pytest
import fast 
import numpy

P = None
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


# Test config parsing and data checking 
def test_config_default():
    c = fast.conf.ConfigParser("test/test_params.py")

def test_config_bounds():
    c = fast.conf.ConfigParser("test/test_params.py")
    with pytest.raises(ValueError):
        c.config['W0'] = -1
    
def test_config_allowed():
    c = fast.conf.ConfigParser("test/test_params.py")
    with pytest.raises(ValueError):
        c.config['NPXLS'] = "hello"
    
def test_config_max_len():
    c = fast.conf.ConfigParser("test/test_params.py")
    with pytest.raises(ValueError):
        c.config['DTHETA'] = [1,2,3]

# Test complete simulation
def test_sim_default():
    sim = fast.Fast("test/test_params.py")
    sim.run()
    global P
    P = sim.params.copy()
    assert numpy.isfinite(sim.result.power).all()
    assert numpy.isfinite(sim.result.dB_rel).all()
    assert numpy.isfinite(sim.result.dB_abs).all()

def test_sim_mean_irradiance():
    sim = fast.Fast("test/test_params.py")
    psf = sim.compute_mean_irradiance()
    assert numpy.isfinite(psf.all())

@pytest.mark.skipif(not fast.fast._pyfftw, reason="No pyfftw installed")
def test_sim_fftw():
    p = P.copy()
    p['FFTW'] = True
    run_sim(p)

def test_sim_randomScrns():
    p = P.copy()
    p['TEMPORAL'] = False
    run_sim(p)

def test_sim_subharm():
    p = P.copy()
    p['SUBHARM'] = True
    p['TEMPORAL'] = False
    run_sim(p)

def test_sim_obsc():
    p = P.copy()
    p['OBSC_GROUND'] = 0.1
    run_sim(p)

def test_sim_obsc_sat():
    p = P.copy()
    p['OBSC_SAT'] = 0.05
    run_sim(p)

def test_sim_axicon():
    p = P.copy()
    p['W0'] = 0.1
    p['AXICON'] = True
    p['OBSC_GROUND'] = 0.1
    run_sim(p)

def test_sim_L_SAT():
    p = P.copy()
    p['L_SAT'] = 500e3
    sim = fast.Fast(p)
    assert sim.L == p['L_SAT']

def test_sim_L0():
    p = P.copy()
    p['L0'] = 25
    run_sim(p)

def test_sim_down():
    p = P.copy()
    p['PROP_DIR'] = 'down'
    run_sim(P)

def test_sim_NOAO():
    p = P.copy()
    p['AO_MODE'] = 'NOAO'
    run_sim(p)

def test_sim_TT():
    p = P.copy()
    p['AO_MODE'] = 'TT'
    run_sim(p)

def test_sim_noise():
    p = P.copy()
    p['NOISE'] = 1
    run_sim(p)

def test_sim_modal():
    p = P.copy()
    p['MODAL'] = True
    run_sim(p)


# Test comms 
def test_sim_coherent():
    p = P.copy()
    p['COHERENT'] = True
    sim = fast.Fast(p)
    sim.run()
    assert sim.I.dtype == complex

def test_fast_fsoc_ook():
    p = P.copy()
    p['MODULATION'] = "OOK"
    sim = fast.comms.FastFSOC(p)
    sim.run()
    assert numpy.isfinite(sim.I).all()
    assert hasattr(sim.modulator, "sep")
    assert numpy.isfinite(sim.modulator.sep)
    assert numpy.isfinite(sim.modulator.evm)

def test_fast_fsoc_BPSK():
    p = P.copy()
    p['MODULATION'] = "BPSK"
    sim = fast.comms.FastFSOC(p)
    sim.run()
    assert numpy.isfinite(sim.I).all()
    assert hasattr(sim.modulator, "sep")
    assert numpy.isfinite(sim.modulator.sep)
    assert numpy.isfinite(sim.modulator.evm)

def test_fast_fsoc_qam():
    p = P.copy()
    p['MODULATION'] = "QAM"
    sim = fast.comms.FastFSOC(p)
    sim.run()
    assert numpy.isfinite(sim.I).all()
    assert hasattr(sim.modulator, "sep")
    assert numpy.isfinite(sim.modulator.sep)
    assert numpy.isfinite(sim.modulator.evm)

# def test_fast_fsoc_ook_withdata():
#     data = numpy.random.normal(size=100).tobytes()
#     p = p.copy()
#     sim = fast.Fast(p)
#     sim.run()
#     mod = fast.comms.Modulator(sim.result.power, 'OOK', 10, data=data, symbols_per_iter=len(data))
#     mod.run()
#     assert numpy.isfinite(sim.recv_symbols)

def test_ber_ook():
    p = P.copy()
    sim = fast.Fast(p)
    sim.run()
    ber = fast.comms.ber_ook(10, sim.result.power)

def test_ber_ook_nosamples():
    ber = fast.comms.ber_ook(10)
    assert numpy.isfinite(ber)

def test_sep_qam():
    p = P.copy()
    sim = fast.Fast(p)
    sim.run()
    ber = fast.comms.ber_qam(4, 10, samples=sim.result.power)
    assert numpy.isfinite(ber)

def test_ber_qam_nosamples():
    ber = fast.comms.ber_qam(4, 10)
    assert numpy.isfinite(ber)

# Test complete orbit simulation
def test_complete_orbit_simulation():
    import test_satellite_param

    p = P.copy()
    p_orbit = test_satellite_param.p_simu.copy()
    tle_path = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle'

    simulation_test = None
    simulation_test = fast.complete_orbit_simulation.FAST_sat_orbit(p, p_orbit, tle_path)
    assert simulation_test[list(simulation_test.keys())[0]]