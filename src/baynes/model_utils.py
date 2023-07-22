import glob
import io
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanModel


def set_models_path(path: str) -> None:
    """
    Validate, then set the Stan models directory path.
    """
    if not os.path.isdir(path):
        raise ValueError(f"No CmdStan directory, path {path} does not exist.")
    os.environ["STAN_MODELS_DIR"] = path
    print(
        " Path to models directory set, add \n",
        f'export STAN_MODELS_DIR="{path}" \n',
        "to .bashrc to make the change permanent.",
    )


def get_models_path() -> str:
    """
    Validate, then return the Stan models directory path.
    """
    models_dir = ""
    if "STAN_MODELS_DIR" in os.environ and len(os.environ["STAN_MODELS_DIR"]) > 0:
        models_dir = os.environ["STAN_MODELS_DIR"]
    else:
        raise ValueError(
            'Path to the models directory not set, use "baynes.model_utils.set_model_path(path)"'
        )

    if not os.path.isdir(models_dir):
        raise ValueError(f"No CmdStan directory, path {models_dir} does not exist.")
    return os.path.normpath(models_dir)


def get_stan_file(stan_file: str) -> str or None:
    """
    Return a .stan file from the models directory path.
    """
    models_path = get_models_path()
    files = glob.glob(get_models_path() + "/**/" + stan_file, recursive=True)
    if len(files) == 0:
        raise ValueError(
            f"File {stan_file} not found in models directory {models_path}."
        )
    elif len(files) > 1:
        print(f"Found multiple files in directory {models_path}.")
        return select_from_list(files)
    else:
        print("Found .stan file ", files[0])
        return files[0]


def select_from_list(options):
    """
    User selection from list of options
    """
    print(f"Select an option number [1-{str(len(options))}]:")
    for idx, option in enumerate(options):
        print(str(idx + 1) + ") " + str(option))
    inputValid = False
    while not inputValid:
        inputRaw = input()
        inputNo = int(inputRaw) - 1
        if inputNo > -1 and inputNo < len(options):
            selected = options[inputNo]
            print(f"Selected: {str(selected)}")
            inputValid = True
            break
        else:
            print("Please select a valid option number")

    return selected


def inits_from_priors(model, prior_fit, n_chains, dir="inits"):
    if not os.path.isdir(dir):
        os.makedirs(dir)

    stan_parameters = model.src_info()["parameters"].keys()
    n_samples = prior_fit._iter_sampling
    init_files = []

    for i in range(n_chains):
        file_name = dir + "/init" + str(i) + ".json"
        init_files.append(file_name)
        idx = np.random.randint(0, n_samples)
        inits = {}
        for par in stan_parameters:
            inits[par] = prior_fit.stan_variable(par)[idx]
            if isinstance(inits[par], np.ndarray):
                inits[par] = inits[par].tolist()
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(inits, f, ensure_ascii=False, indent=4)
    return init_files

