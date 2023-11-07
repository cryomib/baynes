# baynes
Tools and models for bayesian data analysis.

## Dependecies
* This package works for `python >= 3.8`. It is strongly suggested to create a dedicated python environment, for example using `virtualenv`, which can be installed with pip
   ```
   pip install virtualenv
   ```

* Automatic installation of `CmdStan` requires `g++>=4.9.3` and `make>=3.81`

## Installation

Open the terminal and clone the repository locally:

   use a password protected SSH key
   ```
   git clone git@github.com:cryomib/baynes.git
   ```
   or Clone with HTTPS using the web URL
   ```
   git clone https://github.com/cryomib/baynes.git
   ```
### Automatic install
Move into the repository and run the install script
   ```
   cd baynes
   source install.sh
   ```
This will automatically execute all the steps of the manual installation using a default configuration.
### Manual install
1. Create the baynes virtual environment based on `python3.X (X>=8)` and activate it.
   ```
   pip install virtualenv
   virtualenv -p `which python3.X` baynesenv
   source baynesenv/bin/activate
   ```
   (optional) Check the version of python being used in the virtual environment
   ```
   (baynesenv) python -V
   ```

2. Install the required Python packages and dependencies, which include CmdStanPy. With the virtual environment active, move into the `baynes` folder and install the package with pip.
   ```
   (baynesenv) cd baynes
   (baynesenv) pip install .
   ```
   (optional)

3. Install CmdStan and Stan. If `g++>=4.9.3` and `make>=3.81` are already present, this can be done automatically using CmdStanPy's built-in function `install_cmdstan`:
   ```
   (baynesenv) install_cmdstan
   ```
   By default, this will install CmdStan and Stan's core utilities in `$HOME/.cmdstan`. For more informations, see https://mc-stan.org/cmdstanpy/installation.html

   If you want a custom CmdStan installation, follow https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html

   Note: do not follow the conda installation procedure.

4. (optional)  `baynes` allows to retrieve the Stan models from a user defined directory and set the default Stan compiler options. Run:
   ```
   (baynesenv) python scripts/set_defaults.py
   ```
   This will set `baynes/stan/` as the models' base folder and add `baynes/stan/include/` to the compiler search path. Otherwise, the arguments specified in `baynes/config.json` will be used as defaults. Additional default values can be set by updating the config dictionary with dedicated functions found in `baynes/model_utils.py`.

5. (optional) install the jupyter kernel to make `baynesenv` available in notebooks
   ```
   (baynesenv) ipython kernel install --user --name=baynesenv
   ```
   Note: Jupyter notebooks must be run after deactivating the environment.

## Project top-level directory layout

    baynes
    │
    ├── baynes                         # Project source code
    ├── examples                       # Jupyter Notebooks demonstrating various models and techniques
    ├── scripts                        # Simple python scripts for configuration
    ├── stan                           # Collection of tested Stan models and functions
    ├── requirements.txt               # Requirements file specifing the python packages to install
    ├── setup.py                       # Package installation script
    ├── install.sh                     # Full automatic installation script
    ├── .gitignore                     # Specifies intentionally untracked files to ignore
    └── README.md                      # README file

 ASCII art tree structure taken from [here](https://codepen.io/patrickhlauke/pen/azbYWZ)
