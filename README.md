# baynes
Tools for bayesian data analysis
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

### Installation
1. Open the terminal and clone the repository locally:

   use a password protected SSH key (preferred)
   ```
   $ git clone git@github.com:cryomib/baynes.git
   ```
   or Clone with HTTPS using the web URL
   ```
   $ git clone https://github.com/cryomib/baynes.git
   ```

2. Install virtualenv using `python >= 3.8`. Create the baynes virtual environment based on `python3.X (X>=8)` and activate it.

   ```
   $ sudo pip install virtualenv
   $ virtualenv -p `which python3.X` ~/baynesenv
   $ source baynesenv/bin/activate
   ```
   (optional) Check the version of python being used in the virtual environment
   ```
   (baynesenv) $ python -V
   ```

3. Install the required Python packages and dependencies, which include CmdStanPy. With the virtual environment active, move into the `baynes` folder and install the package with pip.
   ```
   (baynesenv) $ cd baynes
   (baynesenv) $ pip install .
   ```
   (optional) `baynes` allows to retrieve the Stan models from a user defined directory and set the default Stan compiler options. Run:
   ```
   (baynesenv) $ python scripts/set_defaults.py
   ```
   To set `stan/` as the models' base folder and add `stan/include/` to the compiler search path.

4. Install CmdStan and Stan. If `g++>=4.9.3` and `make>=3.81` are already present, this can be done automatically using CmdStanPy's built-in function `install_cmdstan`:
   ```
   (baynesenv) $ install_cmdstan
   ```
   By default, this will install CmdStan and Stan's core utilities in `$HOME/.cmdstan`. For more informations, see https://mc-stan.org/cmdstanpy/installation.html

   If you want a custom CmdStan installation, follow https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html

   Note: do not follow the conda installation procedure.

### Project top-level directory layout

    baynes
    │
    ├── baynes                         # Project source code
    ├── stan                           # Collection of tested Stan models and functions
    ├── examples                       # Demonstrations and tutorials
    ├── notebooks                      # Jupyter Notebooks for development and testing of new features
    ├── scripts                        # Simple python scripts for configuration
    ├── requirements.txt               # Requirements file specifing the python packages to install
    ├── setup.py                       # Package installation script
    ├── .gitignore                     # Specifies intentionally untracked files to ignore
    └── README.md                      # README file

 ASCII art tree structure taken from [here](https://codepen.io/patrickhlauke/pen/azbYWZ)