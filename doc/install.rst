Installation
============

There is nothing to compile since FAST is pure python, just ensure that the FAST directory is on your ``PYTHONPATH``, or navigate to the directory and run 

    python setup.py install

To confirm correct installation, try running from the FAST directory:

    | cd test 
    | python test_script.py

which should run a short simulation. 

Requirements
____________
- aotools (https://www.github.com/AOtools/aotools)
- numpy
- scipy 
- astropy
- tqdm
- pyfftw (optional, speeds up FFTs)
- sphinx (optional, to build documentation)
