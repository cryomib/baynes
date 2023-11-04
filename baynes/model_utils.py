import glob
import json
import os

import numpy as np
from cmdstanpy import CmdStanModel

import baynes


def _get_config_file():
    """
    Get the path to the Baynes configuration file.

    Returns:
        str: The path to the configuration file.
    """
    baynes_path = os.path.dirname(baynes.__file__)
    config_file = os.path.join(baynes_path, "config.json")
    if not os.path.isfile(config_file):
        raise ValueError("No config.json file found for baynes")
    return config_file


def set_models_path(path: str) -> None:
    """
    Validate, then set the Stan models directory path.

    Parameters:
        path (str): The directory path to the Stan models.
    """
    if not os.path.isdir(path):
        raise ValueError(f"No CmdStan directory, path {path} does not exist.")

    cfile = _get_config_file()
    with open(cfile, "r") as f:
        config = json.load(f)

    config["STAN_MODELS_DIR"] = path

    with open(cfile, "w") as f:
        json.dump(config, f, indent=4)


def get_models_path() -> str:
    """
    Validate, then return the Stan models directory path.

    Returns:
        str: The directory path to the Stan models.
    """
    with open(_get_config_file(), "r") as f:
        config = json.load(f)
    models_dir = ""
    if "STAN_MODELS_DIR" in config.keys() and len(config["STAN_MODELS_DIR"]) > 0:
        models_dir = config["STAN_MODELS_DIR"]
    else:
        raise ValueError(
            'Path to the models directory not set, use "baynes.model_utils.set_model_path(path)"'
        )
    if not os.path.isdir(models_dir):
        raise ValueError(
            f"No CmdStan directory, path {models_dir} does not exist.")
    return os.path.normpath(models_dir)


def get_config() -> dict:
    """
    Retrieve the configuration file.

    Returns:
        dict: Dictionary containing baynes' configuration.
    """
    with open(_get_config_file(), "r") as f:
        config = json.load(f)
    return config


def update_config(new_config: dict) -> None:
    """
    Update the configuration file.

    Parameters:
        compiler_kwargs (dict): Dictionary containing config keywords and arguments.
    """
    cfile = _get_config_file()
    with open(cfile, "r") as f:
        config = json.load(f)

    config.update(new_config)

    with open(cfile, "w") as f:
        json.dump(config, f, indent=4)


def get_stan_file(stan_file: str) -> str or None:
    """
    Return a .stan file from the models directory path.

    Parameters:
        stan_file (str): The name of the .stan file to find.

    Returns:
        str or None: The path to the .stan file or None if not found.
    """

    if not stan_file.endswith(".stan"):
        stan_file += ".stan"
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


def get_model(stan_file: str) -> CmdStanModel or None:
    """
    Return a stan file model the models directory path, compiling with default arguments

    Parameters:
        stan_file (str): The name of the .stan file to find.

    Returns:
        CmdstanModel or None: The model corresponding to the .stan file or None if not found.
    """
    return CmdStanModel(stan_file=get_stan_file(stan_file), **get_config()['STAN_COMPILER_KWARGS'])


def select_from_list(options):
    """
    Allow the user to select an option from a list.

    Parameters:
        options (list): List of options to choose from.

    Returns:
        Any: The selected option from the list.
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
    """
    Generate initialization files from prior sampling results.

    This function generates initialization files for Stan chains based on prior sampling results. It takes a Stan model
    object (`model`) and a prior fit object (`prior_fit`) as inputs and creates `n_chains` initialization files with
    random initial values for the Stan parameters.

    Parameters:
        model: The Stan model object.
        prior_fit (CmdStanMCMC): The CmdStanMCMC object obtained from prior sampling.
        n_chains (int): The number of chains to generate.
        dir (str, optional): The directory path to save the initialization files. Default is "inits".

    Returns:
        list: List of paths to the generated initialization files.
    """
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
