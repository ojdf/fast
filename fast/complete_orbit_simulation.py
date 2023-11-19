# Library importation
import numpy 
import fast
import datetime
from skyfield.api import load, wgs84

ts = load.timescale()

def get_satellite_obj(TLE_file_path, satellite_name = None):
    '''
    Function to directly obtain a skyfield satellite object from a TLE and a satellite name (if different TLEs in the file)

    INPUTS:
        TLE_file_path : [string] - path to a local or online TLE file
        satellite_name : [string] - Name of the satellite to look at if different satellites are present in the same file
    
    OUTPUTS:
        satellite : the corresponding skyfield satellite object

    '''
    satellites = load.tle_file(TLE_file_path)
    if satellite_name != None:
        by_name = {sat.name: sat for sat in satellites}
        satellite = by_name[satellite_name]
    else:
        satellite = satellites[0]
    return satellite
    
def get_sample_time(satellite, tele_lat, tele_lon, N=10, start=None, period=10, min_altitude_degrees=5.0, max_altitude_degree=90.0, zenith_stop = False):
    '''
    Function to "sample" a satellite orbit passing over a telescop. 
    Research - on a given period - a moment when the sattelite is visible by the telescop, obtain the rising and falling times, and then samples this last period.
    The function returns the sampled times (in [s]) as a list of int values starting at 0, and the initial time t0 as a datetime object. 

    INPUTS:
        satellite : a skyfield satellite object 
        tele_lat/tele_lon : [float] - latitude and longitude of the telescop
        N : [int] -  Number of samples expected
        start : [datetime object] - starting epoch for the research of a period when the satellite is visible by the telecsop. Current time by default.
        period : [int] - the time range in which the research is proceed (given in [days])
        min_altitude_degrees : [float] - minimum satellite altitude expected 
        max_altitude_degrees : [float] - maximum satellite altitude considered
        zenith_stop : [bool] - if True, the sampling is done between min_altitude_degrees and max_altitude_degrees, instead of min_altitude_degrees/min_altitude_degrees
    
    OUTPUTS:
        sample_times : [int list] - delay between t0 and each sample (in [s])
        t0 : [datetime] - epoch of the first sample, given in UTC reference system.

    '''    
    telescop = wgs84.latlon(tele_lat, tele_lon)
    difference = satellite - telescop
    # Research starting time
    if start != None:
        t0 = ts.from_datetime(start)
    else:
        t0 = satellite.epoch
    
    # Research ending time
    t1 = ts.from_datetime( t0.utc_datetime() + datetime.timedelta(days=period))

    times, events = satellite.find_events(telescop, t0, t1, min_altitude_degrees)
    # Maximum altitude research
    max_alt = 0
    max_idx = None
    for idx in range(len(events)):
        alt, az, dist = difference.at(times[idx]).altaz()
        if (events[idx] == 1) and (max_altitude_degree >= alt.degrees >= max_alt):      
            max_idx = idx
            max_alt = alt.degrees
    if max_idx == None:
        raise Exception("The satellite doesn't pass over the telescop during the research period")
    
    # satellite rise and fall time
    idx = max_idx
    while (idx > 0) and (events[idx] != 0):
        idx -= 1
    t_rise = times[idx]

    if zenith_stop:
        t_fall = times[max_idx]
    else:
        idx = max_idx
        while (idx < len(events)-1) and (events[idx] != 2):
            idx += 1
        t_fall = times[idx]

    dt = (t_fall.utc_datetime() - t_rise.utc_datetime()).seconds

    # Sampling
    sample_times = numpy.linspace(0, dt, N)
    
    return sample_times, t_rise.utc_datetime()


def get_angles_positions(sample_times, satellite, tele_lat, tele_lon, t_rise, Tloop, rotations = False):
    '''
    Function to obtain the Point-Ahead Angle (PAA), or the apparent wind equivalent angle, on each given sample time.

    INPUTS:
        t_rise = [datetime] - epoch of the first sample, in the UTC reference system
        sample_times = [int list] - delay between t_rise and each sample (in [s])
        satellite = a skyfield satellite object 
        tele_lat/tele_lon = [float] - latitude and longitude of the telescop
        Tloop = [float] - AO loop delay (in [s])

    OUTPUTS:
        paa = [float list] - PAAs 
        aniso_dl = [float list] - anisoplanetism angle for the downlink
        altitudes = [float list] - altitude of the satellite on each sample (in [°])
        azimuts = [float list] - azimut of the satellite on each sample (in [°])
        distances = [float list] - distance between the satellite and the telescop on each sample (in [m])
        rotations = [float list] - telescop Field of View rotation on each sample (in [°])
    
    '''
    
    celerity = 2.997925e8
    telescop = wgs84.latlon(tele_lat, tele_lon)
    N_sample = len(sample_times)
    difference = satellite - telescop

    dt_paa = numpy.zeros(len(sample_times))
    dx_paa = numpy.zeros(N_sample)
    dy_paa = numpy.zeros(N_sample)
    dx_dl = numpy.zeros(N_sample)
    dy_dl = numpy.zeros(N_sample)
    altitudes = numpy.zeros(N_sample)
    azimuts = numpy.zeros(N_sample)
    distances = numpy.zeros(N_sample)

    rot = numpy.zeros(N_sample)

    for idx, t in enumerate(sample_times):
        # Calculation of the satellite position, in the telescope reference frame, at t0
        topocentric0 = difference.at(ts.from_datetime(datetime.timedelta(seconds=t) + t_rise))
        alt0, az0, dist0 = topocentric0.altaz()
        altitudes[idx], azimuts[idx], distances[idx] = alt0.degrees, az0.degrees, dist0.m
        
        # Calculation of the position of the point ahead angle
        dt_paa[idx] = 2*dist0.m/celerity
        past_telescop = wgs84.latlon(tele_lat, tele_lon - 360*dt_paa[idx]/(24*3600))
        diff_paa = satellite - past_telescop
        topocentric_paa = diff_paa.at(ts.from_datetime(datetime.timedelta(seconds=t+dt_paa[idx]) + t_rise))
        alt_paa, az_paa, dist_paa = topocentric_paa.altaz()

        # Calculation of the position of the satellite at t0+tau_loop
        topocentric_aniso_dl = difference.at(ts.from_datetime(datetime.timedelta(seconds=t+Tloop) + t_rise))
        alt_dl, az_dl, dist_dl = topocentric_aniso_dl.altaz()

        # Calculation of the PAA components in the telescope FoV frame
        cos_alpha_paa = numpy.cos(numpy.pi/2 - alt_paa.radians)*numpy.cos(numpy.pi/2 - alt0.radians) + numpy.sin(numpy.pi/2 - alt_paa.radians)*numpy.sin(numpy.pi/2 - alt0.radians)*numpy.cos(az_paa.radians-az0.radians)
        sin_alpha_paa = numpy.sqrt(1 - cos_alpha_paa**2)
        cos_orientation_paa = (numpy.cos(numpy.pi/2 - alt_paa.radians) - cos_alpha_paa*numpy.cos(numpy.pi/2 - alt0.radians))/(sin_alpha_paa*numpy.sin(numpy.pi/2 - alt0.radians))
        sin_orientation_paa = numpy.sqrt(1 - cos_orientation_paa**2)
        dy_paa[idx] = cos_orientation_paa * numpy.arccos(cos_alpha_paa)*360/(2*numpy.pi)
        dx_paa[idx] = numpy.sign(az_paa.degrees - az0.degrees) * sin_orientation_paa * numpy.arccos(cos_alpha_paa)*360/(2*numpy.pi)

        # Calculation of the downlink anisoplanetism angle components in the telescope FoV frame
        cos_alpha_dl = numpy.cos(numpy.pi/2 - alt_dl.radians)*numpy.cos(numpy.pi/2 - alt0.radians) + numpy.sin(numpy.pi/2 - alt_dl.radians)*numpy.sin(numpy.pi/2 - alt0.radians)*numpy.cos(az_dl.radians-az0.radians)
        sin_alpha_dl = numpy.sqrt(1 - cos_alpha_dl**2)
        cos_orientation_dl = (numpy.cos(numpy.pi/2 - alt_dl.radians) - cos_alpha_dl*numpy.cos(numpy.pi/2 - alt0.radians))/(sin_alpha_dl*numpy.sin(numpy.pi/2 - alt0.radians))
        sin_orientation_dl = numpy.sqrt(1 - cos_orientation_dl**2)
        dy_dl[idx] = cos_orientation_dl * numpy.arccos(cos_alpha_dl)*360/(2*numpy.pi)
        dx_dl[idx] = numpy.sign(az_dl.degrees - az0.degrees) * sin_orientation_dl * numpy.arccos(cos_alpha_dl)*360/(2*numpy.pi)

        # Calculation of the FoV rotation - to be removed
        if rotations:
            beta_0 = numpy.arccos((numpy.cos(numpy.pi/2 - alt_dl.radians) - numpy.cos(numpy.pi/2 - alt0.radians)*cos_alpha_dl)/(sin_alpha_dl*numpy.sin(numpy.pi/2 - alt0.radians)))
            beta_1 = numpy.arccos((numpy.cos(numpy.pi/2 - alt0.radians) - cos_alpha_dl*numpy.cos(numpy.pi/2 - alt_dl.radians))/(sin_alpha_dl*numpy.sin(numpy.pi/2 - alt_dl.radians)))
            rot[idx] =  numpy.pi - beta_1 - beta_0

    

    paa = (numpy.array([dx_paa, dy_paa])*3600).T
    paa[:, 0] = [0 if numpy.isnan(paa[i, 0]) else paa[i, 0] for i in range(len(paa[:, 0]))]
    paa[:, 1] = [0 if numpy.isnan(paa[i, 1]) else paa[i, 1] for i in range(len(paa[:, 1]))]

    aniso_dl = (numpy.array([dx_dl, dy_dl])*3600).T
    aniso_dl[:, 0] = [0 if numpy.isnan(aniso_dl[i, 0]) else aniso_dl[i, 0] for i in range(len(aniso_dl[:, 0]))]
    aniso_dl[:, 1] = [0 if numpy.isnan(aniso_dl[i, 1]) else aniso_dl[i, 1] for i in range(len(aniso_dl[:, 1]))]

    if rotations:
        return paa, aniso_dl, altitudes, azimuts, distances, rot
        
    return paa, aniso_dl, altitudes, azimuts, distances  # Translation_angles in arcsec / rotations in rad
    

def FAST_sat_orbit(fast_params, simu_params, TLE_file):
    '''
    Function that samples an satellite orbit - while this satellite is passing over a telescop - and generate a FAST simualtion object for each sample.

    INPUTS :
        fast_params = [dic] - parameters for the fast simulation
        simu_params = [dic] - complementary parameters associated to this simulation
        TLE_fil = [string] - path to a local or online TLE file
        cn2_turb = [float list] - Cn^2 profile ( in [m^1/3])
        wind_speed = [float list] - wind speed profile (in [m/s])
        wind_dir = [float list] - wind direction profile (in [°])
        h_turb = [float list] - height associated to the previous profiles (in [m])

    OUTPUTS :
        sampled_fast_sim = [dic] - a dictionary containing all the FAST simulation objects and the list of the associated altitudes. The dictionary keays are wrote : 'simulation_idx', with idx in [0, N-1]
    
    '''
    fast_param_cpy = fast_params.copy()

    # Sampled data
    satellite = get_satellite_obj(TLE_file, simu_params['satellite_name'])
    sample_times, t0 = get_sample_time(satellite, simu_params['telescop_lat'], simu_params['telescop_lon'], simu_params['N_sample'], simu_params['t0_research'], simu_params['research_window'], simu_params['altitude_min'], simu_params['altitude_max'], simu_params['zenith_stop'])
    PAAs, aniso_dl, altitudes, azimuts, distances = get_angles_positions(sample_times, satellite, simu_params['telescop_lat'], simu_params['telescop_lon'], t0, fast_param_cpy['TLOOP'])
    zenital_angles = 90 - altitudes

    # FAST simulation object
    layer_mask = numpy.array(fast_params['CN2_TURB'])>0
    fast_param_cpy['CN2_TURB'] = numpy.array(fast_params['CN2_TURB'])[layer_mask]
    fast_param_cpy['H_TURB'] = numpy.array(fast_params['H_TURB'])[layer_mask]
    sampled_fast_sim = {}
    for idx,theta_z in enumerate(zenital_angles):
        fast_param_cpy['L_SAT'] = distances[idx]
        fast_param_cpy['DTHETA'] = PAAs[idx, :]
        fast_param_cpy['ANISO_DL'] = aniso_dl[idx, :]
        fast_param_cpy['ZENITH_ANGLE'] = theta_z
        fast_param_cpy['AZIMUT_SAT'] = azimuts[idx]

        fast_param_cpy['WIND_DIR'] = (numpy.array(fast_params['WIND_DIR']))[layer_mask]
        fast_param_cpy['WIND_SPD'] = (numpy.array(fast_params['WIND_SPD']))[layer_mask]

        simu = fast.Fast(fast_param_cpy)
        sampled_fast_sim[f'simulation_{idx}'] = simu
    
    sampled_fast_sim['altitudes'] = altitudes
    
    return sampled_fast_sim

def FAST_sat(sat_apparent_speed, fast_params):
    fast_params['ANISO_DL'] = sat_apparent_speed*fast_params['TLOOP']
    return fast.Fast(fast_params)
