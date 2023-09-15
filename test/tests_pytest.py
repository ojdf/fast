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
    ptmp['W0'] = 0.1
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

def test_fast_fsoc_ook():
    ptmp = p.copy()
    ptmp['MODULATION'] = "OOK"
    sim = fast.comms.FastFSOC(ptmp)
    sim.run()
    assert numpy.isfinite(sim.I).all()
    assert hasattr(sim.modulator, "sep")
    assert numpy.isfinite(sim.modulator.sep)
    assert numpy.isfinite(sim.modulator.evm)

def test_fast_fsoc_BPSK():
    ptmp = p.copy()
    ptmp['MODULATION'] = "BPSK"
    sim = fast.comms.FastFSOC(ptmp)
    sim.run()
    assert numpy.isfinite(sim.I).all()
    assert hasattr(sim.modulator, "sep")
    assert numpy.isfinite(sim.modulator.sep)
    assert numpy.isfinite(sim.modulator.evm)

def test_fast_fsoc_qam():
    ptmp = p.copy()
    ptmp['MODULATION'] = "QAM"
    sim = fast.comms.FastFSOC(ptmp)
    sim.run()
    assert numpy.isfinite(sim.I).all()
    assert hasattr(sim.modulator, "sep")
    assert numpy.isfinite(sim.modulator.sep)
    assert numpy.isfinite(sim.modulator.evm)

# def test_fast_fsoc_ook_withdata():
#     data = numpy.random.normal(size=100).tobytes()
#     ptmp = p.copy()
#     sim = fast.Fast(ptmp)
#     sim.run()
#     mod = fast.comms.Modulator(sim.result.power, 'OOK', 10, data=data, symbols_per_iter=len(data))
#     mod.run()
#     assert numpy.isfinite(sim.recv_symbols)

def test_ber_ook():
    ptmp = p.copy()
    sim = fast.Fast(ptmp)
    sim.run()
    ber = fast.comms.ber_ook(10, sim.result.power)

def test_ber_ook_nosamples():
    ber = fast.comms.ber_ook(10)
    assert numpy.isfinite(ber)

def test_sep_qam():
    ptmp = p.copy()
    sim = fast.Fast(ptmp)
    sim.run()
    ber = fast.comms.ber_qam(4, 10, samples=sim.result.power)
    assert numpy.isfinite(ber)

def test_ber_qam_nosamples():
    ber = fast.comms.ber_qam(4, 10)
    assert numpy.isfinite(ber)