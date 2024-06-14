Configuration
=============

FAST accepts a number of configuration parameters, which must be defined in a python dictionary ``p`` contained in a ``.py`` file. An example configuration file is shown below. See the comment for each configuration parameter for a description. 

Some things to keep in mind:

- Some parameters can be set to ``"auto"`` or ``"opt"``, in which case the simulation will *attempt* to select parameters that will provide good results. This is not a guarantee that the parameters will be optimal for your particular setup!
- FAST expects atmospheric turbulence profiles in units of :math:`C_n^2 (h) \, \mathrm{d}h`, i.e. **integrated** Cn2 for each layer, units m :sup:`1/3`
- Some analytical atmopsheric turbulence models (HV57, Bufton wind model) are provided with useful functions in :py:mod:`fast.turbulence_models`

.. literalinclude:: ../test/test_params.py

