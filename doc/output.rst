Running and Output
==================

Once a config file has been created, the simulation is initialised by creating a :py:class:`~fast.fast.Fast` instance with your config file

    | import fast
    | sim = fast.Fast("your_config.py")

On initialisation, most of the relevant analytical values are automatically computed, including the relevant power spectra and link budget terms. The slow(er) Monte Carlo aspect of the simulation is started by calling :py:func:`~fast.fast.Fast.run`

    | sim.run()

This will iterate over random or temporally correlated atmopsheric realisations, computing the coupled flux or relevant metric either at the satellite (uplink) or ground (downlink). A progress bar provides an indication of how long the process will take.

Once completed, the simulation results can be found in ``sim.result`` (also returned from ``sim.run()``). This :py:class:`~fast.fast.FastResult` class allows rapid conversion to useful units. Currently, these are:

- ``result.dB_rel`` - results in dB relative to the diffraction limit (no turbulence). 
- ``result.dB_abs`` - results in dB including all link budget terms, i.e. Rx power / Tx power
- ``result.power`` - results in linear power, units of Watts

Simulation results can be saved to FITS file with most simulation parameters as header information by calling the :py:func:`~fast.fast.Fast.save` function on the sim class

    | sim.save("save_filename.fits")

    