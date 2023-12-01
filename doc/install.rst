Installation
============

Install from PyPI with

``pip install fast-sim``

For the latest releases, the source code can be obtained from https://github.com/ojdf/fast. There is nothing to compile since FAST is pure python, just ensure that the FAST directory is on your ``PYTHONPATH``, or navigate to the directory and run 

    | pip install -r requirements.txt
    | pip install .

To confirm correct installation, try running from the FAST directory:

    | cd test 
    | python test_script.py

which should run a short simulation. 

Requirements
____________
.. include:: ../requirements.txt
   :literal:

Unit Testing Requirements
_________________________
.. include:: ../test/requirements.txt
    :literal:

Optional Requirements
_____________________
- pyfftw (speeds up FFTs)
- sphinx (build documentation)