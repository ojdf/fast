import numpy

def HV57(h, w=21, A=1.7e-14):
    return 0.00594 * (w/27)**2 * (1e-5 * h)**10 * numpy.exp(-h/1000) + 2.7e-16 * numpy.exp(-h/1500) + A * numpy.exp(-h/100.)

def Bufton_wind(h, zenith_angle=0, vg=8, vt=30, ht=9400., Lt=4800.):
    return vg + vt * numpy.exp(-((h * numpy.cos(numpy.radians(zenith_angle))-ht)/Lt)**2)