import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict


class MatplotlibHelper:
    def __init__(self, col_wrap=3, fig_scale=6, save=False, output_dir='figures/', output_format='.jpeg', style='ggplot'):
        plt.style.use(style)
        self.style = style
        self.figures = {}
        self.current_title = None
        self.col_wrap = col_wrap
        self.fig_scale = fig_scale
        self.output_dir = output_dir
        self.set_savefig(save)
        self.format = output_format

    def set_savefig(self, save):
        if save:
            self.set_output_dir(self.output_dir)
        self.save = save

    def set_output_dir(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def get_figure_title(self, partial_title=None):
        if partial_title == 'all':
            titles = list(self.figures)
        elif partial_title is None:
            titles = [self.current_title]
        else:
            titles = [title for title in list(self.figures) if partial_title in title]
        return titles

    def get_current_figure(self):
        return self.figures[self.current_title]

    def generate_new_title(self, plot_type):
        n_figures_with_type = sum(plot_type in title for title in list(self.figures))
        return plot_type + '_' + str(n_figures_with_type)

    def new_figure(self, plot_type, figure=None):
        if figure is None:
            figure = plt.figure()
        figure_title = self.generate_new_title(plot_type)
        self.figures[figure_title] = figure
        self.current_title = figure_title
        return figure

    def clear_figures(self, titles=None, clear_files=True):
        titles = self.get_figure_title(partial_title=titles)
        for title in titles:
            if title in self.figures:
                del self.figures[title]
                figure_path = f"{self.output_dir}{title}{self.format}"
                if clear_files and os.path.exists(figure_path):
                    os.remove(figure_path)

    def save_figures(self, titles=None):
        titles = self.get_figure_title(partial_title=titles)
        for title in titles:
            if title in self.figures:
                self.figures[title].savefig(f"{self.output_dir}{title}{self.format}", bbox_inches='tight')

    def add_lines(self, x_coords=[], y_coords=[], label=None, bbox_to_anchor=(1.05, 0.6), facecolor='white', edgecolor='white', **kwargs):
        fig = self.get_current_figure()
        axes = fig.axes
        for i, x in enumerate(x_coords):
            axes[i].axvline(x, label=label, **kwargs)
        for i, y in enumerate(y_coords):
            axes[i].axhline(y, label=label, **kwargs)

        if label is not None:
            self.update_legend(fig=fig, last_labels=[label], bbox_to_anchor=bbox_to_anchor, facecolor=facecolor, edgecolor=edgecolor)
        return fig

    def update_legend(self, fig=None, last_labels=None, **lgd_kws):
        if fig is None:
            fig = self.get_current_figure()
        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        for lgd in fig.legends:
            h = lgd.legendHandles
            l = [t.get_text() for t in lgd.get_texts()]
            handles.extend(h)
            labels.extend(l)
            lgd.remove()
        by_label = OrderedDict(zip(labels, handles))
        if last_labels is not None:
            for l in last_labels:
                by_label.move_to_end(l)
        fig.legend(by_label.values(), by_label.keys(), **lgd_kws)
  
    def resize(self, x, y):
        fig = self.get_current_figure()
        fig.set_size_inches(x, y)
        return fig


class FitPlotter(MatplotlibHelper):
    
    def __init__(self, fit=None, fit_title=None, **kwargs):
        self.fits = {}
        if fit is not None:
            self.add_fit(fit, fit_title)
        super().__init__(**kwargs)

    def add_fit(self, fit, fit_title=None, rep_names=None):
        if isinstance(fit_title, str) is False:
            fit_title = 'fit' + str(len(self.fits))
        self.fits[fit_title] = fit
        self.current_fit = fit_title
        method_variables = list(fit.method_variables().keys())
        all_variables = [p for p in fit.column_names]
        if rep_names is None:
            rep_names = ['_rep']
        rep_variables = [col for col in all_variables if any(
            gen in col for gen in rep_names)]
        stan_variables = [
            var for var in all_variables if var not in method_variables + rep_variables]
        self.method_variables = method_variables
        self.rep_variables = rep_variables
        self.stan_variables = stan_variables

    def get_fit(self, title=None):
        if title is None:
            title = self.current_fit
        return self.fits[title]

    def draws_df(self, fit_titles=None, parameters=None, inc_warmup=False):
        if fit_titles is None:
            fit_titles = [self.current_fit]
        elif fit_titles == 'all':
            fit_titles = list(self.fits.keys())

        draws_df = pd.DataFrame([])
        for fit_n in fit_titles:
            draws_temp = self.fits[fit_n].draws_pd(
                inc_warmup=inc_warmup)[parameters]
            draws_temp['fit'] = fit_n
            draws_df = pd.concat([draws_df, draws_temp])
        return draws_df

    def validate_parameters(self, params):
        if isinstance(params, list) is False:
            if params == 'all_stan':
                params = self.stan_variables
            else:
                params = [params]
        return params
    
    @staticmethod
    def single_fit_plot(func):
        def wrapper(self, parameters='all_stan', **kwargs):
            parameters = self.validate_parameters(parameters)
            n_plots = len(parameters)
            n_cols = min(n_plots, self.col_wrap)
            n_rows = int(np.ceil(n_plots/n_cols))
            fig = self.new_figure(func.__name__)
            fig.set_size_inches(self.fig_scale*n_cols, self.fig_scale*n_rows*0.7)
            if n_plots == 1:
                subfigs = np.array([fig])
            else:
                subfigs = fig.subfigures(n_rows, n_cols)
                subfigs = subfigs.flatten()
            for i, par in enumerate(parameters):
                subfig = func(self, par, subfigs[i], **kwargs)
            handles, labels = subfig.axes[0].get_legend_handles_labels()
            lgd = fig.legend(handles, labels, bbox_to_anchor=(
                1.15, 0.6), facecolor='white', edgecolor='white')
            for line in lgd.get_lines():
                line.set_linewidth(1.5)
            plt.show()
            if self.save:
                fig.savefig(
                    f"{self.output_dir}{self.current_title}{self.format}", bbox_inches='tight')

        return wrapper

    @staticmethod
    def multi_fit_plot(func):
        def wrapper(self, parameters='all_stan', fit_titles=None, inc_warmup=False, df=None, **kwargs):
            parameters = self.validate_parameters(parameters)
            if df is None:
                draws_df = self.draws_df(
                fit_titles, parameters, inc_warmup=inc_warmup)
            else:
                draws_df = df
            plot = func(self, draws_df, parameters, **kwargs)
            self.new_figure(func.__name__, plot.figure)
            plt.show()
            if self.save:
                plot.figure.savefig(
                    f"{self.output_dir}{self.current_title}{self.format}", bbox_inches='tight')

        return wrapper

    @single_fit_plot
    def convergence_plot(self, par, figure, fit_name=None, alpha=0.9, linewidth=0.3, initial_steps=50, **kwargs):
        ax = figure.subplots(1, 2, width_ratios=[
                             2.5, 1], sharey=True, gridspec_kw={'wspace': 0.1})
        fit = self.get_fit(fit_name)
        n_chains = fit.chains
        draws_df = fit.draws_pd(inc_warmup=True)[[par]]
        if n_chains == 1:
            label = "Chain 1"
        else:
            label = ["Chain "+str(i+1) for i in range(n_chains)]

        draws = np.array(np.array_split(draws_df[par], n_chains))
        draws = np.swapaxes(draws, 0, 1)
        ax[0].plot(draws, label=label, alpha=alpha, linewidth=linewidth, **kwargs)
        ax[0].set_xlabel("step")
        ax[0].set_ylabel(par)
        if fit.metadata.cmdstan_config['save_warmup']:
            ax[0].axvspan(0, fit.num_draws_warmup, color='gray',
                          alpha=0.4, lw=0, label='Warmup')
        ax[0].set_xlim(0)

        ax[1].plot(draws[:initial_steps, :], label=label, alpha=alpha, linewidth=0.5+linewidth, **kwargs)
        ax[1].set_xlabel("initial step")
        ax[1].set_xlim(0)
        return figure

    @single_fit_plot
    def predictive_check(self, rep_key, figure, data=None, data_key=None, fit_name=None, percs=[5, 95], color='green', lines = False, n_bins=None):            
        ax, ax1 = figure.subplots(2, 1, height_ratios=[2.5, 1], sharex=True)
        draws = self.get_fit(fit_name).draws_pd(rep_key, inc_warmup=False).to_numpy()
        events = np.array(data[data_key])
        if n_bins is not None:
            events, bins = np.histogram(events, bins=n_bins)
            draws = np.array([np.histogram(dr, bins=bins)[0] for dr in draws])

        ax.plot(events, color=color, linewidth=1.5, label=data_key, linestyle='--')
        ax1.plot(np.zeros(len(events)), color=color, linewidth=1.5, linestyle='--', label=data_key)

        if lines:
            for i in range(min(80, len(draws))):
                if i==0:
                    ax.plot(draws[i], color=color, linewidth=0.3, alpha=0.4, label=rep_key)
                else:
                    ax.plot(draws[i], color=color, linewidth=0.3, alpha=0.4)
                ax1.plot((draws[i]-events)/np.sqrt(events), color=color, linewidth=0.3, alpha=0.5)
        else:
            lo, hi = np.percentile(draws, percs, axis=0)
            ax.fill_between(np.arange(len(events)), lo, hi,
                            color=color, alpha=0.4, label=rep_key)
            ax1.fill_between(np.arange(len(events)), (lo-events)/np.sqrt(events), (hi-events)/np.sqrt(events),
                        color=color, alpha=0.4, label=rep_key)
        if n_bins is not None:
            ax.set_ylabel('counts')    
            ax1.set_xlabel(data_key)         
        else:
            ax.set_ylabel(data_key)    
            ax1.set_xlabel('bin')  
        ax1.set_ylabel('residuals')

        return figure

    @multi_fit_plot
    def pair_grid(self, df, parameters, hue='fit', height=1.5, corner=True, **kwargs):
        grid = sns.PairGrid(df, hue=hue, height=height, corner=corner)
        grid.map_offdiag(sns.scatterplot, **kwargs)
        grid.map_diag(sns.histplot, bins=20)
        return grid

    @multi_fit_plot
    def dis_plot(self, df, parameters, legend=True, facet_kws = {"sharey": False, "sharex": False, 'legend_out':True}, kind = "kde", hue = 'fit', col='variable', **kwargs):
        dmelt = df.melt(id_vars=['fit'])
        displot = sns.displot(data=dmelt, x='value', legend=legend, facet_kws=facet_kws,
                              kind=kind, hue=hue, col=col, col_wrap=min(len(parameters), self.col_wrap), height=self.fig_scale/1.5, **kwargs)
        for i, ax in enumerate(displot.axes.flatten()):
            ax.set_xlabel(parameters[i])
        displot.set_titles("")
        return displot
    
    @multi_fit_plot
    def cat_plot(self, df, parameters, id_vars=['fit'], legend = True, sharex = False, kind = 'box', hue=None, col = 'variable', **kwargs):
        dmelt = df.melt(id_vars=id_vars)
        catplot = sns.catplot(data=dmelt, x='value', y='fit', legend = legend, sharex = sharex, 
                              kind = kind, hue=hue, col = col, col_wrap=min(len(parameters), self.col_wrap), height=self.fig_scale/1.5, **kwargs)
        catplot.set_titles("")
        for i, ax in enumerate(catplot.axes.flatten()):
            ax.set_xlabel(parameters[i])
            ax.set_ylabel('')

        return catplot
    
    @multi_fit_plot
    def kde_plot(self, df, parameters, hue='fit', **kwargs):
        def informative_kde(x=None, percs=[5, 50, 95], color='purple', median_color='black', label=None, bw_adjust=0.3):
            ax = sns.kdeplot(x, fill=False, bw_adjust=bw_adjust, color=color, label=label)
            kdeline = ax.lines[-1]
            x_data = kdeline.get_xdata()
            y_data = kdeline.get_ydata()
            left, middle, right = np.percentile(x, percs)
            ax.vlines(middle, 0, np.interp(middle, x_data, y_data),
                      color=median_color, ls=':', linewidth=1.5, label='median')
            ax.fill_between(x_data, 0, y_data,
                            color=color, alpha=0.4, label=str(percs[0])+"-"+str(percs[2])+'%')
            ax.fill_between(x_data, 0, y_data, where=(left <=x_data) & (x_data <= right), interpolate=False, color=color, alpha=0.5, lw=.2)
            return ax

        dmelt = df.melt(id_vars=['fit'])        
        kdegrid = sns.FacetGrid(dmelt, col='variable', hue=hue, col_wrap=min(len(parameters), self.col_wrap), sharey=False, sharex=False, height=self.fig_scale/1.5)
        kdegrid.map(informative_kde, 'value', **kwargs)
        kdegrid.add_legend(label_order= kdegrid.hue_names + ['median'], title='',bbox_to_anchor=(1.05, 0.5)) 
        kdegrid.set_titles("")
        for i, ax in enumerate(kdegrid.axes.flatten()):
            ax.set_xlabel(parameters[i])
        return kdegrid

    @multi_fit_plot
    def ridgeplot(self, df, parameters, row='fit', col='variable', hue='fit', pcolor=-3):
        n_fits = df.fit.nunique()
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        def mykde(x=None, color='purple', label=None):
            ax = sns.kdeplot(x, clip_on=False, fill=False, alpha=1, linewidth=1.2, color="w", bw_adjust=.3)
            kdeline = ax.lines[0]
            x_data = kdeline.get_xdata()
            y_data = kdeline.get_ydata()
            median = np.percentile(x, [50])
            ax.vlines(median, 0, np.interp(median, y_data,y_data), color="orange", ls=':')
            ax.fill_between(x_data, 0, y_data, color=color, alpha=1)

        dmelt = df.melt(id_vars=['fit'])
        pal = sns.cubehelix_palette(n_fits+3, rot=-.4, start=pcolor, reverse=True)
        grid = sns.FacetGrid(dmelt, row=row, hue=hue, col=col, aspect=self.fig_scale, height=1, palette=pal, sharex=True, sharey=False)

        grid.map(mykde, 'value')
        grid.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        def label(x, color, label):
            ax = plt.gca()
            ax.text(-0.1, 0.1, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)
        grid.map(label, 'value')
        grid.figure.subplots_adjust(hspace=-0.75)
        grid.set_titles("")
        grid.set(yticks=[], ylabel="")
        grid.set_xlabels(parameters[0])
        grid.despine(bottom=True, left=True)
        plt.style.use(self.style)
        return grid

