# baynes
Tools for bayesian data analysis
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

### Installation
1. Open the terminal and clone the repository locally:
   Clone using a password protected SSH key (preferred)
   ```
   $ git clone git@github.com:cryomib/baynes.git
   ```
   or Clone with HTTPS using the web URL
   ```
   $ git clone https://github.com/cryomib/baynes.git
   ```

2. Install virtualenv using `python >= 3.8`
   ```
   $ sudo pip install virtualenv
   ```
   Create the baynes virtual environment based on python3.X (X>=8)
   ```
   $ virtualenv -p `which python3.X` ~/baynesenv
   ```
   Run the following command to activate the virtual environment
   ```
   $ source baynesenv/bin/activate
   ```
   (optional) Check the version of python being used in the virtual environment
   ```
   (baynesenv) $ python -V
   ```

3. Install the required Python packages and dependencies, which include CmdStanPy. With the virtual environment active, move into the `baynes` folder
   ```
   (baynesenv) $ cd baynes
   ```
   and type
   ```
   (baynesenv) $ pip install .
   ```

4. Install CmdStan and Stan. If `g++>=4.9.3` and `make>=3.81` are already present, this can be done automatically using CmdStanPy's built-in function `install_cmdstan`:
   ```
   (baynesenv) $ install_cmdstan
   ```
   By default, this will install CmdStan and Stan's core utilities in '$HOME/.cmdstan'. For more informations, see https://mc-stan.org/cmdstanpy/installation.html

   If you want a custom CmdStan installation, follow https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html

   Note: do not follow the conda installation procedure.

### Project top-level directory layout

    baynes
    │
    ├── src                            # Project source code
    ├── scripts                        # Directory for scripts and executables
    ├── test                           # Directory used to collect test code
    ├── requirements.txt  # Requirements file specifing the lists of packages to install
    ├── setup.py                       #
    ├── .gitignore                     # Specifies intentionally untracked files to ignore
    └── README.md                      # README file

 ASCII art tree structure taken from [here](https://codepen.io/patrickhlauke/pen/azbYWZ)

### Documentation
*

 ### About README
 The README files are text files that introduces and explains a project. It contains information that is commonly required to understand what the project is about.
 At this [link](https://help.github.com/en/github/writing-on-github/basic-writing-and-formatting-syntax) a basic guide on the writing and formatting syntax
