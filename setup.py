from distutils.core import setup


with open("README.md", "r") as readme:
    long_description= readme.read()


setup(
    name='fast',
    author_email='o.j.d.farley@durham.ac.uk',
    url='https://github.com/ojdf/fast',
    packages=['fast'],

    description='Fourier domain Adaptive Optics Simulation Tool',
    long_description=long_description,
    version='0.1'
)