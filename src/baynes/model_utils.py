import glob
import io
import json
import os
import re
import subprocess

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pygraphviz as pgv
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


def remove_comments(code_str):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, code_str)


def get_code_blocks(code_str, block_names="all"):
    if block_names == "all":
        block_names = [
            "functions",
            "data",
            "transformed_data",
            "parameters",
            "transformed parameters",
            "model",
            "generated quantities",
        ]
    model_dict = {}
    lines = code_str.split("\n")
    block_start = None
    for i, line in enumerate(lines):
        if block_start is None:
            for name in block_names:
                if line.startswith(name):
                    block_start = i + 1
                    block_name = name
                    break
        else:
            if line.startswith("}"):
                block_end = i - 1
                block_content = "\n".join(lines[block_start : block_end + 1])
                model_dict[block_name] = block_content
                block_start = None
    return model_dict


def get_sampling_statements(model_str):
    sampling_statements = []
    for line in re.split(r"{|}|;", model_str):
        if "~" in line or "target +=" in line:
            sampling_statements.append(line.strip())
    return sampling_statements


def parse_sampling_statements(statement_list):
    stan_file = ".temp_model_only.stan"
    with open(stan_file, "w") as outfile:
        outfile.write("model{\n")
        outfile.write("\n".join(str(i) + ";" for i in statement_list))
        outfile.write("}")

    parsed_raw = subprocess.run(
        ["stanc", "--debug-parse", stan_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.remove(stan_file)
    parsed_string = parsed_raw.stdout.decode("utf-8")
    parsed_string = parsed_string.replace("Parser: ", "")
    parsed_string = re.split(r"tilde_statement|targetpe_statement", parsed_string)
    samplings = []
    for statement in parsed_string:
        samplings.append([])
        for line in statement.splitlines():
            if "identifier " in line:
                samplings[-1].append(line.replace("identifier ", ""))
        if len(samplings[-1]) == 0:
            del samplings[-1]
    return samplings


def get_model_graph(model, plot=True, extended=False):
    if isinstance(model, str):
        model = CmdStanModel(stan_file=model, compile=False)

    code = get_code_blocks(model.code(), block_names=["model"])["model"]
    samplings = parse_sampling_statements(get_sampling_statements(code))
    info = model.src_info()
    all_dist = [dist.split("_")[0] for dist in info["distributions"]]
    all_pars = list(info["parameters"].keys()) + list(
        info["transformed parameters"].keys()
    )
    if extended:
        all_data = list(info["inputs"].keys()) + list(
            info["transformed parameters"].keys()
        )
        all_pars += all_data

    G = pgv.AGraph(directed=True)

    for statement in samplings:
        L = len(statement)
        if L >= 3:
            if statement[0] in all_dist:
                dist, var = statement[0:2]
            else:
                var, dist = statement[0:2]
            for i in range(2, L):
                if statement[i] in all_pars:
                    if extended:
                        G.add_edge(statement[i], var, label=dist)
                    else:
                        G.add_edge(statement[i], var)

    if plot:
        img_bytes = G.draw(format="png", prog="dot")
        img = mpimg.imread(io.BytesIO(img_bytes), format="png")
        plt.imshow(img)
        plt.axis("off")

    return G
