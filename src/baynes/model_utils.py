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


def get_model_graph(model, plot=True):
    if isinstance(model, str):
        model = CmdStanModel(stan_file=model, compile=False)

    code = get_code_blocks(model.code(), block_names=["model"])["model"]
    samplings = parse_sampling_statements(get_sampling_statements(code))
    info = model.src_info()
    all_pars = list(info['parameters'].keys())+list(info['transformed parameters'].keys())
    all_data = list(info['inputs'].keys())+list(info['transformed parameters'].keys())
    all_dist = [dist.split("_")[0] for dist in info["distributions"]]
    G = pgv.AGraph(directed=True)

    for statement in samplings:
        L = len(statement)
        if L >= 3:
            if statement[0] in all_dist:
                dist, var = statement[0:2]
            else:
                var, dist = statement[0:2]
            for i in range(2, L):
                if statement[i] in all_data + all_pars:
                    G.add_edge(statement[i], var, label=dist)

    if plot:
        img_bytes = G.draw(format="png", prog="dot")
        img = mpimg.imread(io.BytesIO(img_bytes), format="png")
        plt.imshow(img)
        plt.axis("off")

    return G
