import pickle
from multiprocessing.dummy import Pool

import matplotlib.pyplot as plt

from baynes.model_utils import inits_from_priors


def multithreaded_run(function, args, n_processes, filename=None):
    pool = Pool(processes=n_processes)
    results = pool.map(function, args)
    if filename is not None:
        output = open(filename, "wb")
        pickle.dump(results, output)
        output.close()
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
    if auto_prior_var in model.src_info()["inputs"].keys():
        print("\n ---- Sampling the priors ---- \n")
        data[auto_prior_var] = 1
        sampler_kwargs["show_progress"] = False
        fit_prior = model.sample(data, **sampler_kwargs)

        print("\n ---- Prior predictive check ---- \n")
        plotter.add_fit(fit_prior, fit_title=fit_title + "_prior")
        plotter.predictive_check(rep_key, data=data, data_key=data_key)

        print("\n ---- Prior distribustions ---- \n")
        plotter.dis_plot( plot_params, kind="hist", hue="variable", common_bins=False, element="step", alpha=0.7, lw=1.5)
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

    print("\n ---- Posterior distribustions ---- \n")
    plotter.dis_plot( plot_params, kind="hist", hue="variable", common_bins=False, element="step", alpha=0.7, lw=1.5)
    plt.show()

    print("\n ---- Prior vs posterior comparison ---- \n")
    plotter.cat_plot(parameters=plot_params, fit_titles=[fit_title, fit_title + "_prior"])
    return fit
