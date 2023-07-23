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
    author               = 'Pietro Campana, Matteo BorgheSsi',
    author_email         = 'campana.pietro@gmail.com, matteo.borghesi@mib.infn.it',
    url                  = 'https://github.com/cryomib/baynes',
#    license              = read('LICENSE'),
    install_requires     = read('requirements.txt').splitlines(),
    package_dir          = {'': 'baynes'},
    packages             = find_packages(where='baynes'),
    package_data         = {'': ['config.json']},
    include_package_data = True,
    python_requires      = '>=3.8',
    zip_safe             = False,
)