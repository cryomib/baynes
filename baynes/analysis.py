import pickle
from multiprocessing.dummy import Pool
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
    fit_title="fit",
    rep_key="counts_rep",
    data_key="counts",
    plot_params="all_stan",
    auto_prior_var="prior",
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
        plotter.predictive_check(rep_key, data=data, data_key=data_key)

        print("\n ---- Prior distributions ---- \n")
        plotter.dis_plot(plot_params, kind="hist", hue="variable", common_bins=False, element="step", alpha=0.7, lw=1.5)
        sampler_kwargs["inits"] = inits_from_priors(
            model, fit_prior, sampler_kwargs["chains"]
        )
        plt.show()
        data[auto_prior_var] = 0

    sampler_kwargs["show_progress"] = True
    print("\n ---- Fitting the model ---- \n")
    fit = model.sample(data, **sampler_kwargs)
    print(fit.diagnose())

    plotter.add_fit(fit, fit_title=fit_title)
    plotter.convergence_plot(parameters=plot_params, initial_steps=100)

    print("\n ---- Prior predictive check ---- \n")
    plotter.predictive_check(rep_key, data=data, data_key=data_key)

    print("\n ---- Posterior distributions ---- \n")
    plotter.dis_plot(plot_params, kind="hist", hue="variable", common_bins=False, element="step", alpha=0.7, lw=1.5)
    plt.show()

    print("\n ---- Prior vs posterior comparison ---- \n")
    plotter.cat_plot(parameters=plot_params, fit_titles=[fit_title, fit_title + "_prior"])
    return fit
