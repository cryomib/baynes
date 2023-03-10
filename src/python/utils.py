import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['lines.linewidth'] = 0.9
#matplotlib.rcParams['axes.prop_cycle'] = cycler('color')
sns.set_style('whitegrid')
sns.set_palette("Set1",plt.cm.tab20c.N )



def convergence_plot(fit, pars=[], first_steps=30):
    config = fit.metadata.cmdstan_config
    n_chains = config['num_chains']
    warmup = config['save_warmup']
    if warmup:
        n_warm = config['draws_warmup']/config['thin']
    df = fit.draws_pd(inc_warmup=warmup)
    params = list(fit.stan_variables().keys())+list(fit.method_variables().keys())
    for p in pars:
        if p in params:
            if n_chains==1:
                label = "Chain 1"
            else:
                label = ["Chain "+str(i+1) for i in range(n_chains)]
            draws=np.array(np.array_split(df[p], n_chains))
            draws = np.swapaxes(draws,0,1)
            fig, (ax,ax1)=plt.subplots(1,2,width_ratios=[2.5,1], figsize=(8,4), sharey=True )
            ax.plot(draws, label = label, alpha=0.9, linewidth=0.3);
            ax.set_xlabel("step")
            ax.set_ylabel(p)
            if warmup:
                ax.axvspan(0, n_warm, color='gray', alpha=0.2, lw=0, label='Warmup')

            lgd = ax.legend(ncol=np.floor(n_chains/3)+1)
            for line in lgd.get_lines():
                line.set_linewidth(1.5)
            #(loc = 'upper center', bbox_to_anchor = (0.38, 1.05), markerscale = 10, ncol=np.floor(n_chains/2)+1)
            ax1.plot(draws[:first_steps,:], label = label, alpha=0.9, linewidth=1.5 );
            ax1.set_xlabel("initial step")
            fig.tight_layout()
        else:
            print(p, " is not a valid parameter name, try:\n", params)



def ridgeplot(df, x, var_name, nvals, pcolor=-3):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(nvals+3, rot=-.4, start=pcolor, reverse=True)
    g = sns.FacetGrid(df, row=var_name, hue=var_name, aspect=8, height=0.8,palette=pal)

    def mykde(x, color, label):
        ax=sns.kdeplot(x, clip_on=False, fill=False, alpha=1, linewidth=1.2, color="w", bw_adjust=.3)
        kdeline = ax.lines[0]
        mean = x.mean()
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        height = np.interp(mean, xs, ys)

       # boxplot(x, ax)
       # ax.vlines(mean, 0, height, ls=':',color="green")
       # ax.fill_between(xs, 0, ys, alpha=1, color=color)

        left, middle, right = np.percentile(x, [25, 50, 75])
        ax.vlines(middle, 0, np.interp(middle, xs, ys), color="orange", ls=':')
       # ax.vlines(left, 0, np.interp(left, xs, ys), color="orange", ls=':')
        #ax.vlines(right, 0, np.interp(right, xs, ys), color="orange", ls=':')

        ax.fill_between(xs, 0, ys, color=color, alpha=1)
        ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=False, color=color, alpha=0.5, lw=.2)



    # Draw the densities in a few steps
    #g.map(sns.kdeplot, x, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    #g.map(sns.histplot, x, clip_on=False, color='w', lw=2)#bw_adjust=.3,

    # passing color=None to refline() uses the hue mapping
    g.map(mykde, x)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.6)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_palette("Set1",plt.cm.tab20c.N )




def var_progression(model, data, var, var_name, var_f=None, chains=1, samples=3000, warm=1000, keys='all'):
    if var_name not in list(data.keys()):
        print('Not a model variable, try: \n', list(data.keys()))
    else:
        res={var_name:[]}
        for x in var:
            data_temp = data.copy()
            if var_f is None:
                data_temp[var_name] = x
            else:
                var_f(data_temp, x)
            fit = model.sample(data=data_temp,
                               chains=chains,
                               parallel_chains=chains,
                               iter_warmup=warm,
                               iter_sampling=samples,
                               save_warmup=False,
                               show_progress=False)

            df_temp = fit.draws_pd(inc_warmup=False)
            if keys=='all':
                keys= list(fit.stan_variables().keys())
            for key in keys:
                if key not in list(res.keys()):
                    res[key]=[]

                res[key]+=list(df_temp[key])
            res[var_name]+=[var_name+" = "+str(x)]*samples*chains
                #print(res)
        return pd.DataFrame.from_dict(res)


def check_hist(data, draws, n_reps=50, color='green'):
    tdf=draws.transpose()
    sns.histplot(data, discrete=True, color=color, element="poly", fill=False, linewidth=1.5)
    for i in range(n_reps):
        sns.histplot(tdf[i], discrete=True, color=color,element="poly", fill=False, linewidth=0.5, alpha=0.2)


def predictive_check(model, data, data_keys, fit=None, var=None, rep_keys=None, n_iter=1000, n_warm=0, n_plot=50, prior=False):

    if rep_keys is None:
        rep_keys = [k+'_rep' for k in data_keys]

    if fit is None:
        print('No fit found, start sampling\n')
        fit = model.sample(data=data,
                           chains=1,
                           iter_warmup=n_warm,
                           iter_sampling=n_iter,
                           save_warmup=False,
                           show_progress=False,
                           fixed_param=prior,
                           output_dir='./output')

    for i in range(len(data_keys)):
        df = fit.draws_pd(rep_keys[i])
        check_hist(data[data_keys[i]], df, n_reps=n_plot);

    if var is not None:
        vdf = fit.draws_pd(var)
        #for v in var:
        plt.figure()
        sns.histplot(vdf);
    return fit

def posterior_kde(data, ax=None,  percs=[5, 50, 95], color='purple', label=None):
    if ax is None:
        ax = sns.kdeplot(data, fill=False, bw_adjust=.3, color=color)
    else:
        sns.kdeplot(data, fill=False, bw_adjust=.3, color=color, ax=ax)


    kdeline=ax.lines[0]
    mean = data.mean()
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(mean, xs, ys)


    left, middle, right = np.percentile(data, percs)

    ax.vlines(middle, 0, np.interp(middle, xs, ys), color="orange", ls=':', linewidth=3, label='median')
    ax.vlines(mean, 0, np.interp(mean, xs, ys), color="lightgreen", ls=':', linewidth=3, label='mean')
    ax.fill_between(xs, 0, ys, color=color, alpha=0.4, label=str(percs[0])+"-"+str(percs[2])+'%')
    ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=False, color=color, alpha=0.5, lw=.2)
    if label is True:
        ax.legend()
