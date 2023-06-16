# FAST (Fourier domain Adaptive optics Simulation Tool)

![](https://github.com/ojdf/fast/actions/workflows/unit_test.yml/badge.svg)
[![codecov](https://codecov.io/gh/ojdf/fast/branch/main/graph/badge.svg?token=Z1U8JBPE48)](https://codecov.io/gh/ojdf/fast)

FAST is a simulation tool that utilises a Fourier domain AO model to enable rapid Monte Carlo characterisation of free space optical links between the Earth and satellites. For more details, see the paper [https://doi.org/10.1364/OE.458659](https://doi.org/10.1364/OE.458659).

WARNING: FAST is under active development and the simulation API and configuration is not fixed. If you find a bug, or something is not working as you would expect, please email me or make an issue on GitHub, it's likely I've broken something! 

## Requirements
See `requirements.txt`

It is recommended to install [pyFFTW](https://pypi.org/project/pyFFTW/) to speed up FFTs. 

## Installation
FAST is pure python, so hopefully there is not anything to compile, just ensure that the FAST directory is on your `PYTHONPATH`, or navigate to the directory and run 

`python setup.py install`

To confirm correct installation, try running 

`cd test && python test_script.py`

in the FAST directory, which should run a short simulation. 

## Configuration
Config is handled currently by python scripts, an example of which is shown in `demo_params.py`. The config file should define a dictionary `p` which contains all of the configuration information required. If any values are missed, they will be replaced by defaults. 

To start a simulation either in the python shell or in a script, you can either pass the filename of the script defining the config dictionary, or you can provide the dictionary itself, which can be useful if you are scanning through parameters. So
```
import fast
sim = fast.Fast("your_config_file.py")
```
or 
```
import fast
p = {your config}
sim = fast.Fast(p)
```
## Running the simulation 
The simulation is run by calling the `run()` function on the sim object. This will compute the phase screens and log-amplitude values, and then compute the detected power or phase/amplitude for coherent detection. The results are stored in `sim.result` and are also returned from `run()`, i.e. `res = sim.run()`. Results in either dB or power units can be obtained from e.g. `sim.result.dB_rel` (for dB relative to the diffraction limit) or `sim.result.power`. See the documentation for more details.

To save the simulation results as a fits file with header information, use `sim.save(filename)`.