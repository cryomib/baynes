import os
from setuptools import setup, find_packages

PACKAGE = "baynes"

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name                 = PACKAGE,
    version              = '0.0.1',
    description          = 'Tools for bayesian data analysis using STAN',
    long_description     = read('README.md'),
    author               = 'Pietro Campana, Matteo Borghesi',
    author_email         = 'campana.pietro@gmail.com, matteo.borghesi@mib.infn.it',
    url                  = 'https://github.com/cryomib/baynes',
    install_requires     = read('requirements.txt').splitlines(),
    packages             = find_packages(exclude=[".github", "stan", "examples", "scripts"]),
    package_data         = {'baynes': ['config.json']},
    include_package_data = True,
    python_requires      = '>=3.8',
    zip_safe             = False,
)