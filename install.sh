#! /bin/bash
virtualenv ../baynesenv
source ../baynesenv/bin/activate
pip install .
install_cmdstan --progress #--cores $(nproc --all)
ls -F ~/.cmdstan
python scripts/set_defaults.py
ipython kernel install --user --name=baynesenv
