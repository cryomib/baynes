import pickle
import json
import os
import numpy as np
from multiprocessing.dummy import Pool
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from baynes.model_utils import inits_from_priors

def multithreaded_run(function, args, n_processes, filename=None):
    """
    Run the given function in parallel using multiple threads.

    Parameters:
        function (callable): The function to be executed in parallel.
        args (list): A list of arguments to be passed to the function.
        n_processes (int): The number of threads (processes) to use for parallel execution.
        filename (str, optional): If provided, the results will be pickled and saved in this file.

    Returns:
        list: List containing the results of function calls for each input argument.
    """
    pool = Pool(processes=n_processes)
    results = pool.map(function, args)
    if filename is not None:
        try:
            with open(filename, "wb") as output:
                pickle.dump(results, output)
        except IOError as e:
            print(f"Error saving results to file: {e}")
    pool.close()
    pool.join()

    return results

def standard_analysis(
    model,
    data,
    plotter,
    sampler_kwargs,
    output_dir=None,
    fit_title="fit",
    plot_params="all_stan",
    auto_prior_var="prior",
    rep_key="counts_rep",
    data_key="counts",
    **pcheck_kwargs
):
    """
    Perform a standard analysis for Bayesian modeling.

    This function performs a standard analysis for Bayesian modeling, including prior predictive checks, fitting the model,
    and generating posterior distributions. It can also handle auto-generated priors if specified.

    Parameters:
        model: The Stan model object.
        data (dict): The data dictionary for the model.
        plotter: The plotter object to visualize the results.
        sampler_kwargs (dict): Keyword arguments for Stan's sampling method.
        fit_title (str, optional): Title for the fit results. Default is "fit".
        rep_key (str, optional): Replicated variable for predictive checks. Default is "counts_rep".
        data_key (str, optional): Data variable for predictive checks. Default is "counts".
        plot_params (str or list, optional): Parameters to plot. Default is "all_stan".
        auto_prior_var (str, optional): If provided, it specifies the auto-generated prior variable.
        **pcheck_kwargs: additional keyword arguments or the predictive check plots

    Returns:
        fit: The Stan fit object obtained after fitting the model.
    """
    if auto_prior_var in model.src_info()["inputs"].keys():
        print("\n ---- Sampling the priors ---- \n")
        data[auto_prior_var] = 1
        sampler_kwargs["show_progress"] = False
        fit_prior = model.sample(data, **sampler_kwargs)

        print("\n ---- Prior predictive check ---- \n")
        plotter.add_fit(fit_prior, fit_title=fit_title + "_prior")
        plotter.predictive_check(rep_key, data=data, data_key=data_key, **pcheck_kwargs)

        print("\n ---- Prior distributions ---- \n")
        plotter.dis_plot(plot_params, kind="hist", hue="variable", common_bins=False, element="step", alpha=0.7, lw=1.5)
        if 'inits' not in sampler_kwargs.keys():
            sampler_kwargs["inits"] = inits_from_priors(
                model, fit_prior, sampler_kwargs["chains"])
        plt.show()
        data[auto_prior_var] = 0
    else:
        fit_prior = None

    sampler_kwargs["show_progress"] = True
    print("\n ---- Fitting the model ---- \n")
    fit = model.sample(data, **sampler_kwargs)
    print(fit.diagnose())

    plotter.add_fit(fit, fit_title=fit_title)
    plotter.convergence_plot(parameters=plot_params, initial_steps=100)

    print("\n ---- Posterior predictive check ---- \n")
    plotter.predictive_check(rep_key, data=data, data_key=data_key, **pcheck_kwargs)

    print("\n ---- Posterior distributions ---- \n")
    plotter.dis_plot(plot_params, kind="hist", hue="variable", common_bins=False, element="step", alpha=0.7, lw=1.5)
    plt.show()

    print("\n ---- Prior vs posterior comparison ---- \n")
    plotter.cat_plot(parameters=plot_params, fit_titles=[fit_title, fit_title + "_prior"])
    if output_dir is not None:
        print("\n Saving files to ", output_dir)
        save_analysis(output_dir, data=data, prior=fit_prior, posterior=fit)
    return fit

def sensitivity_sweep(model, data, data_sweep, parameters, n_processes=8, sampler_kwargs={'chains': 1, 'show_progress': False}):
    keys, values = zip(*data_sweep.items())
    permutations = [dict(zip(keys, v)) for v in it.product(*values)]
    all_data = []

    for p in permutations:
        copy = data.copy()
        copy.update(p)
        all_data.append(copy)

    def sample(data):
        fit = model.sample(data, **sampler_kwargs)
        df = fit.draws_pd(parameters)
        for key in keys:
            df[key] = data[key]
        return df

    return pd.concat(multithreaded_run(sample, all_data, n_processes=n_processes))

def save_analysis(dir_path, data=None, prior=None, posterior=None):
    if not os.path.isdir(dir_path):
        raise ValueError(f"Path {dir_path} does not exist.")
    if data is not None:
        dict_to_json(data, os.path.join(dir_path, 'data.json'))
    if prior is not None:
        prior_dir = os.path.join(dir_path, 'prior')
        os.makedirs(prior_dir, exist_ok=True)
        prior.save_csvfiles(prior_dir)
    if posterior is not None:
        posterior_dir = os.path.join(dir_path, 'posterior')
        os.makedirs(posterior_dir, exist_ok=True)
        posterior.save_csvfiles(posterior_dir)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def dict_to_json(dd, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(dd, json_file, indent=4, cls=NumpyEncoder)
    print(f"Dictionary saved to {file_path}")