import json
import os

import numpy as np


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
