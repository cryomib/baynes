# baynes
Tools for bayesian data analysis
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

### Installation
1. Open up your terminal and clone the repository locally:  
   Clone by using a password protected SSH key (preferred)
   ```
   $ git clone git@github.com:cryomib/baynes.git
   ```
   or Clone with HTTPS by using by the web URL
   ```
   $ git clone https://github.com/cryomib/baynes.git
   ```
   
2.  Install virtualenv using python >= 3.8
   ```
   $ sudo pip install virtualenv
   ```
   Create the baynes virtual environment based on python3.X (X>=8) 
   ```
   $ virtualenv -p `which python3.X` ~/baynesenv
   ```   
   Run the following command to activate the baynes virtual environment 
   ```
   $ source baynesenv/bin/activate
   ```  
   (optional) Check the version of python being used in the virtual environment  
   ```
   (baynesenv) $ python -V
   ```
   Python 3.X.x
   if the returned value is the chosen verson, the version is ok

3. Install the packages and the dependencies of baynes inside virtualenv
   Inside the baynes folder, just type
   ```  
   (baynesenv) $ pip install .
   ```  

4. Install CmdStan and Stan
   ```  
   (baynesenv) $ install_cmdstan
   ```  
   Automatically installation as reported in Note 1. It will install CmdStan in $HOME/.cmdstan. If you want a custom CmdStan installation, follow https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html

   Note 1: you can use the build-in function of CmdStan to install Stan (see install_cmdstan function https://mc-stan.org/cmdstanpy/installation.html) 

   Note 2: do not follow the conda installation procedure.

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
