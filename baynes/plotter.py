import os
import pickle
import numpy as np
import itertools as it
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from cmdstanpy import CmdStanMCMC

class MatplotlibHelper:
    def __init__(self, col_wrap=3, fig_scale=5, fig_ratio=1.2, save=False, output_dir='figures/', output_format='.jpeg'):
        """
        Helper class for creating and managing matplotlib figures.

        Args:
            col_wrap (int): Number of columns to wrap the figures.
            fig_scale (float): Base figure size.
            fig_ratio (float): Lenght to height ratio for the figures.
            save (bool): Whether to save the figures.
            output_dir (str): Directory path to save the figures.
            output_format (str): Output format for the saved figures.
        """
        self.figures = {}
        self._previuos_figures = {}
        self.current_title = None
        self.col_wrap = col_wrap
        self.fig_scale = fig_scale
        self.fig_ratio = fig_ratio
        self.output_dir = output_dir
        self.set_savefig(save)
        self.format = output_format

    def set_savefig(self, save):
        """
        Enable or disable saving figures.

        Args:
            save (bool): Whether to save the figures.
        """
        if save:
            self.set_output_dir(self.output_dir)
        self.save = save

    def set_output_dir(self, output_dir):
        """
        Set the output directory for saving figures.

        Args:
            output_dir (str): Directory path to save the figures.
        """
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def get_figure_title(self, partial_title=None):
        """
        Get a list of figure titles.

        Args:
            partial_title (str): Partial title to filter the figure titles.

        Returns:
            list: List of figure titles.
        """
        if partial_title == 'all':
            titles = list(self.figures)
        elif partial_title is None:
            titles = [self.current_title]
        else:
            titles = [title for title in list(self.figures) if partial_title in title]
        return titles

    def get_current_figure(self):
        """
        Get the current figure.

        Returns:
            matplotlib.figure.Figure: Current figure object.
        """
        return self.figures[self.current_title]

    def generate_new_title(self, plot_type):
        """
        Generate a new title for a figure based on the plot type.

        Args:
            plot_type (str): Type of plot.

        Returns:
            str: New figure title.
        """
        n_figures_with_type = sum(plot_type in title for title in list(self.figures))
        return plot_type + '_' + str(n_figures_with_type)

    def new_figure(self, plot_type, figure=None):
        """
        Create a new figure.

        Args:
            plot_type (str): Type of plot.
            figure (matplotlib.figure.Figure, optional): Existing figure object.

        Returns:
            matplotlib.figure.Figure: New or existing figure object.
        """
        if figure is None:
            figure = plt.figure()
        figure_title = self.generate_new_title(plot_type)
        self.figures[figure_title] = figure
        self.current_title = figure_title
        figure.set_size_inches(self.fig_scale*self.fig_ratio, self.fig_scale)
        return figure

    def clear_figures(self, titles=None, clear_files=True):
        """
        Clear the specified figures and optionally delete their saved files.

        Args:
            titles (str or list, optional): Titles of the figures to clear.
            clear_files (bool, optional): Whether to delete the saved figure files.
        """
        titles = self.get_figure_title(partial_title=titles)
        for title in titles:
            if title in self.figures:
                del self.figures[title]
                figure_path = f"{self.output_dir}{title}{self.format}"
                if clear_files and os.path.exists(figure_path):
                    os.remove(figure_path)

    def save_figures(self, titles=None):
        """
        Save the specified figures.

        Args:
            titles (str or list, optional): Titles of the figures to save.
        """
        titles = self.get_figure_title(partial_title=titles)
        for title in titles:
            if title in self.figures:
                self.figures[title].savefig(f"{self.output_dir}{title}{self.format}", bbox_inches='tight')

    def copy(self, item):
        """
        Copy full object with pickle

        Returns:
            copied object
        """
        return pickle.loads(pickle.dumps(item, -1))

    @staticmethod
    def modify_figure(func):
        """
        Allow the current figure state to be recovered before applying a function.
        """
        def wrapper(self, *args, **kwargs):
            plt.close()
            title = self.current_title
            self._previuos_figures[title] = self.copy(self.get_current_figure())
            return func(self, *args, **kwargs)
        return wrapper

    def undo_last(self):
        """
        Undoes the currect figure state before the last applied modify_figure function.

        Returns:
            matplotlib.figure.Figure: Recovered figure object.
        """
        previous = self._previuos_figures
        title = self.current_title
        if title in previous.keys():
            self.figures[title] = previous[title]
            deleted = previous.pop(title, None)
            return self.get_current_figure()
        else:
            print(f'No stored changes for {title}')

    @modify_figure
    def add_lines(self, x_coords=[], y_coords=[], label=None, bbox_to_anchor=(1.05, 0.6), facecolor='white', edgecolor='white', **kwargs):
        """
        Add vertical and horizontal lines to the current figure.

        Args:
            x_coords (list): List of x-coordinates for vertical lines.
            y_coords (list): List of y-coordinates for horizontal lines.
            label (str): Label for the lines.
            bbox_to_anchor (tuple): Bounding box anchor position for the legend.
            facecolor (str): Face color for the legend.
            edgecolor (str): Edge color for the legend.
            **kwargs: Additional keyword arguments to pass to the axvline and axhline functions.

        Returns:
            matplotlib.figure.Figure: Updated figure object.
        """
        fig = self.get_current_figure()
        axes = fig.axes
        for i, x in enumerate(x_coords):
            axes[i].axvline(x, label=label, **kwargs)
        for i, y in enumerate(y_coords):
            axes[i].axhline(y, label=label, **kwargs)

        if label is not None:
            self.update_legend(last_labels=[label], bbox_to_anchor=bbox_to_anchor, facecolor=facecolor, edgecolor=edgecolor)
        return fig

    @modify_figure
    def update_legend(self, last_labels=None, edgecolor='white', bbox_to_anchor=(1.2, 0.8), **lgd_kws):
        """
        Update the legend of the current figure. Allows to change the position of labels and combine legends from different subplots

        Args:
            fig (matplotlib.figure.Figure, optional): Figure object to update the legend.
            last_labels (list, optional): Labels to move to the end of the legend.
            **lgd_kws: Additional keyword arguments for matplotlib.figure.Figure.legend().
        """
        fig = self.get_current_figure()
        handles, labels = [], []
        for ax in fig.axes:
            han, lab = ax.get_legend_handles_labels()
            handles.extend(han)
            labels.extend(lab)
            if ax.legend_ is not None:
                ax.legend_.remove()
        for lgd in fig.legends:
            han = lgd.legendHandles
            lab = [t.get_text() for t in lgd.get_texts()]
            handles.extend(han)
            labels.extend(lab)
            lgd.remove()
        by_label = OrderedDict(zip(labels, handles))
        if last_labels is not None:
            for lab in last_labels:
                by_label.move_to_end(lab)
        fig.legend(by_label.values(), by_label.keys(), edgecolor=edgecolor,
                   bbox_to_anchor=bbox_to_anchor, **lgd_kws)
        return fig

    @modify_figure
    def resize(self, x, y):
        """
        Resize the current figure.

        Args:
            x (float): Width of the figure.
            y (float): Height of the figure.

        Returns:
            matplotlib.figure.Figure: Resized figure object.
        """
        fig = self.get_current_figure()
        fig.set_size_inches(x, y)
        return fig

    def plot(self, *args, **kwargs):
        """
        Simple wrapper for plt.plot

        Args:
            *args: arguments for plt.plot()
            **kwargs: keyword arguments for plt.plot()
        """
        fig = self.new_figure("plot")
        ax = fig.subplots()
        ax.plot(*args, **kwargs)
        plt.show()
        if self.save:
            fig.savefig(
                f"{self.output_dir}{self.current_title}{self.format}", bbox_inches='tight')
        return ax

class FitPlotter(MatplotlibHelper):
    """
    A class for plotting and visualizing Bayesian fit results.

    This class extends the MatplotlibHelper class to provide functionalities for plotting Bayesian fit results.
    It allows adding and managing multiple fits, generating various types of plots, and handling dataframes.

    Parameters:
        fit: The initial fit object to be added (optional).
        fit_title: Title for the initial fit (optional).
        **kwargs: Additional keyword arguments to pass to the MatplotlibHelper constructor.

    Attributes:
        fits (dict): Dictionary containing fit objects with their titles.
        current_fit (str): The title of the currently selected fit.
        method_variables (list): List of variables obtained from the fit's method.
        rep_variables (list): List of replicate variables in the fit.
        stan_variables (list): List of standard variables in the fit.
    """

    def __init__(self, fit=None, fit_title=None, **kwargs):
        self.fits = {}
        if fit is not None:
            self.add_fit(fit, fit_title)
        super().__init__(**kwargs)

    def add_fit(self, fit, fit_title=None, rep_names=None):
        """
        Add a fit to the FitPlotter.

        Parameters:
            fit: The fit object to be added.
            fit_title (str, optional): Title for the fit (default is auto-generated).
            rep_names (list, optional): List of strings for replicate variable names (default is ['_rep']).

        Returns:
            None
        """
        if isinstance(fit_title, str) is False:
            fit_title = 'fit' + str(len(self.fits))
        self.fits[fit_title] = fit
        self.current_fit = fit_title
        if isinstance(fit, CmdStanMCMC):
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
        """
        Get the fit object for the specified title.

        Parameters:
            title (str, optional): The title of the fit to retrieve. If None, the current fit is used.

        Returns:
            The fit object corresponding to the specified title.
        """
        if title is None:
            title = self.current_fit
        return self.fits[title]

    def get_fit_titles(self, title=None):
        if title is None:
            titles = self.current_fit
        elif title == 'all':
            titles = list(self.fits.keys())
        else:
            titles = [t for t in self.fits.keys() if title in t]
        return titles

    def draws_df(self, fit_titles=None, parameters=None, inc_warmup=False):
        """
        Generate a pandas DataFrame containing draws from the specified fits and parameters.

        Parameters:
            fit_titles (list or str, optional): List of fit titles to include in the DataFrame. Default is the current fit.
                                              If 'all', includes all fit titles in reverse order of addition.
            parameters (list or str, optional): List of parameter names to include in the DataFrame.
                                                If 'all_stan', includes all standard variables. Default is None.
            inc_warmup (bool, optional): Whether to include warm-up draws. Default is False.

        Returns:
            pd.DataFrame: DataFrame containing the specified draws.
        """
        if fit_titles is None:
            fit_titles = [self.current_fit]
        elif fit_titles == 'all':
            fit_titles = list(self.fits.keys())[::-1]

        draws_df = pd.DataFrame([])
        for fit_n in fit_titles:
            curr_fit = self.fits[fit_n]
            if isinstance(curr_fit, CmdStanMCMC):
                draws_temp = curr_fit.draws_pd(inc_warmup=inc_warmup)[parameters]
            else:
                draws_temp = curr_fit[parameters]
            draws_temp['fit'] = fit_n
            draws_df = pd.concat([draws_df, draws_temp])
        return draws_df

    def validate_parameters(self, params):
        """
        Validate the parameters for plotting.

        Parameters:
            params (list or str): List of parameter names or a string (e.g., 'all_stan').

        Returns:
            list: List of valid parameter names.
        """
        if isinstance(params, list) is False:
            if params == 'all_stan':
                params = self.stan_variables
            else:
                params = [params]
        else:
            all_pars=self.stan_variables+self.rep_variables
            validated = []
            for p in params:
                if p in all_pars:
                    validated.append(p)
                validated = validated + [par for par in all_pars if p+'[' in par]
            params=validated
        return params

    @staticmethod
    def single_fit_plot(func):
        def wrapper(self, parameters='all_stan', legend=True, **kwargs):
            """
            Wrapper for the plotting function with single-fit capability.

            Parameters:
                parameters (list or str, optional): List of parameter names or a string (e.g., 'all_stan') to be plotted.
                                                    Default is 'all_stan'.
                **kwargs: Additional keyword arguments to pass to the plotting function.

            Returns:
                None
            """
            parameters = self.validate_parameters(parameters)
            n_plots = len(parameters)
            n_cols = min(n_plots, self.col_wrap)
            n_rows = int(np.ceil(n_plots/n_cols))
            fig = self.new_figure(func.__name__)
            if n_plots == 1:
                subfigs = np.array([fig])
                fig.set_size_inches(self.fig_scale*self.fig_ratio, self.fig_scale)
            else:
                subfigs = fig.subfigures(n_rows, n_cols)
                subfigs = subfigs.flatten()
                fig.set_size_inches(self.fig_scale*n_cols*self.fig_ratio, self.fig_scale*n_rows)
               # fig.set_layout_engine('compressed')
            for i, par in enumerate(parameters):
                subfig = func(self, par, subfigs[i], **kwargs)
            if legend:
                handles, labels = subfig.axes[0].get_legend_handles_labels()
                if func.__name__=='predictive_check':
                    lgd = fig.legend(handles, labels, facecolor='white', edgecolor='white',bbox_to_anchor=(
                    0.9, 0.9))
                else:
                    lgd = fig.legend(handles, labels, bbox_to_anchor=(
                    1.2, 0.6), facecolor='white', edgecolor='white')
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
            """
            Wrapper for the plotting function with multi-fit capability.

            Parameters:
                parameters (list or str, optional): List of parameter names or a string (e.g., 'all_stan') to be plotted.
                                                    Default is 'all_stan'.
                fit_titles (list or str, optional): List of fit titles to include in the DataFrame.
                                                    Default is the current fit.
                                                    If 'all', includes all fit titles in reverse order of addition.
                inc_warmup (bool, optional): Whether to include warm-up draws. Default is False.
                df (pd.DataFrame, optional): The DataFrame containing draws for plotting (overrides fit_titles and parameters).

            Returns:
                None
            """
            parameters = self.validate_parameters(parameters)
            if df is None:
                draws_df = self.draws_df(
                fit_titles, parameters, inc_warmup=inc_warmup)
            else:
                draws_df = df
            plot = func(self, draws_df, parameters, **kwargs)
            f = self.new_figure(func.__name__, plot.figure)
            f.set_layout_engine('compressed')

            if func.__name__ != 'pair_grid':
                n_plots = len(parameters)
                n_cols = min(n_plots, self.col_wrap)
                n_rows = int(np.ceil(n_plots/n_cols))

                norm = np.sqrt(2 /(n_rows+n_cols))
                if n_plots > 1:
                    f.set_size_inches(self.fig_scale*self.fig_ratio*n_cols*norm , self.fig_scale*n_rows*norm)
            else:
                f.set_size_inches(self.fig_scale*(len(parameters))**0.3 , self.fig_scale*(len(parameters))**0.3)

            if self.save:
                plot.figure.savefig(
                    f"{self.output_dir}{self.current_title}{self.format}", bbox_inches='tight')

        return wrapper


    @single_fit_plot
    def convergence_plot(self, par, figure, fit_name=None, wspace=0.1, alpha=0.9, linewidth=0.3, initial_steps=50, **kwargs):
        ax = figure.subplots(1, 2, width_ratios=[
                             2.5, 1], sharey=True, gridspec_kw={'wspace': wspace})
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
        """
        Generate a predictive check plot for a single fit.

        Parameters:
            rep_key (str): The key for the replicate variable in the fit draws.
            figure: The Matplotlib figure object to plot on.
            data (pd.DataFrame, optional): Dataframe containing the data used in the fit. Default is None.
            data_key (str, optional): The key for the data column used in the predictive check. Default is None.
            fit_name (str, optional): The title of the fit to use for plotting. Default is the current fit.
            percs (list, optional): List of percentiles for shading the plot. Default is [5, 95].
            color (str, optional): Color of the plot lines and fill. Default is 'green'.
            lines (bool, optional): If True, plot individual lines for each replicate. Default is False.
            n_bins (int, optional): Number of bins to use for histogram-like plots. Default is None.

        Returns:
            figure: The Matplotlib figure object containing the predictive check plot.
        """
        ax, ax1 = figure.subplots(2, 1, height_ratios=[2.5, 1], sharex=True)
        draws = self.get_fit(fit_name).draws_pd(rep_key, inc_warmup=False).to_numpy()
        events = np.array(data[data_key])
        if n_bins is not None:
            events, bins = np.histogram(events, bins=n_bins)
            draws = np.array([np.histogram(dr, bins=bins)[0] for dr in draws])
        std = np.nanstd(draws, axis=0)
        if lines:
            for i in range(min(80, len(draws))):
                if i==0:
                    ax.plot(draws[i], color=color, linewidth=0.4, alpha=0.4, label='replicated')
                else:
                    ax.plot(draws[i], color=color, linewidth=0.4, alpha=0.4)
                ax1.plot((draws[i]-events), color=color, linewidth=0.4, alpha=0.4)
        else:
            lo, hi = np.nanpercentile(draws, percs, axis=0)

            ax.fill_between(np.arange(len(events)), lo, hi,
                            color=color, alpha=0.4, label='replicated')
            ax1.fill_between(np.arange(len(events)), (lo-events), (hi-events),
                        color=color, alpha=0.4)
        ax.plot(events, color='black', linewidth=2, label='observed', linestyle='-')
        ax1.plot(np.zeros(len(events)), color='black', linewidth=1.5, linestyle='--')
        if n_bins is not None:
            ax.set_ylabel('counts')
            ax1.set_xlabel(data_key)
        else:
            ax.set_ylabel(data_key)
            ax1.set_xlabel('bin')
        ax1.set_ylabel('residuals')

        return figure

    @multi_fit_plot
    def pair_grid(self, df, parameters, legend=False, hue='fit', height=1.5, corner=True, s=0.5, **kwargs):
        """
        Generate a pair plot grid for multiple fits.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data for the pair plot.
            parameters (list): List of parameter names to include in the pair plot.
            legend (bool, optional): If True, include a legend. Default is False.
            hue (str, optional): Variable name for coloring the plot by fit. Default is 'fit'.
            height (float, optional): Height of each subplot. Default is 1.5.
            corner (bool, optional): If True, generate only the lower triangle of the grid. Default is True.
            s (float, optional): Marker size for scatterplot. Default is 0.5.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            grid: The Seaborn PairGrid object containing the pair plot grid.
        """
        grid = sns.PairGrid(df, hue=hue, height=height, corner=corner)
        grid.map_lower(sns.scatterplot, s=s, **kwargs)
        grid.map_upper(sns.kdeplot)

        grid.map_diag(sns.histplot, bins=20)
        if legend:
            grid.add_legend(label_order= grid.hue_names, title='',bbox_to_anchor=(0.9, 0.6))
        return grid

    @multi_fit_plot
    def dis_plot(self, df, parameters, legend=True, facet_kws = {"sharey": False, "sharex": False, 'legend_out':True}, kind = "kde", hue = 'fit', col='variable', **kwargs):
        """
        Generate a distribution plot for multiple fits.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data for the distribution plot.
            parameters (list): List of parameter names to include in the distribution plot.
            legend (bool, optional): If True, include a legend. Default is True.
            facet_kws (dict, optional): Additional keyword arguments for facetting the plot. Default is {"sharey": False, "sharex": False, 'legend_out':True}.
            kind (str, optional): Type of the distribution plot (e.g., 'kde', 'hist', 'ecdf'). Default is 'kde'.
            hue (str, optional): Variable name for coloring the plot by fit. Default is 'fit'.
            col (str, optional): Variable name for organizing plots into columns. Default is 'variable'.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            displot: The Seaborn displot object containing the distribution plot.
        """
        dmelt = df.melt(id_vars=['fit'])
        displot = sns.displot(data=dmelt, x='value', legend=legend, facet_kws=facet_kws,
                              kind=kind, hue=hue, col=col, col_wrap=min(len(parameters), self.col_wrap), height=self.fig_scale, **kwargs)
        for i, ax in enumerate(displot.axes.flatten()):
            ax.set_xlabel(parameters[i])
        displot.set_titles("")
        return displot

    @multi_fit_plot
    def cat_plot(self, df, parameters, id_vars=['fit'], x='value', y='fit', legend = True, sharex = False, kind = 'box', hue='fit', col = 'variable', **kwargs):
        """
        Generate a categorical plot for multiple fits.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data for the categorical plot.
            parameters (list): List of parameter names to include in the categorical plot.
            id_vars (list, optional): List of variable names to use as identifier variables. Default is ['fit'].
            x (str, optional): Variable name for the x-axis. Default is 'value'.
            y (str, optional): Variable name for the y-axis. Default is 'fit'.
            legend (bool, optional): If True, include a legend. Default is True.
            sharex (bool, optional): If True, share x-axis across subplots. Default is False.
            kind (str, optional): Type of the categorical plot (e.g., 'box', 'violin', 'bar'). Default is 'box'.
            hue (str, optional): Variable name for coloring the plot. Default is None.
            col (str, optional): Variable name for organizing plots into columns. Default is 'variable'.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            catplot: The Seaborn catplot object containing the categorical plot.
        """
        dmelt = df.melt(id_vars=id_vars)
        catplot = sns.catplot(data=dmelt, legend = legend, sharex = sharex, x=x, y=y,
                              kind = kind, hue=hue, col = col, col_wrap=min(len(parameters), self.col_wrap), height=self.fig_scale, **kwargs)
        catplot.set_titles("")
        for i, ax in enumerate(catplot.axes.flatten()):
            ax.set_xlabel(parameters[i])
            ax.set_ylabel('')

        return catplot

    @multi_fit_plot
    def kde_plot(self, df, parameters, hue='fit', col='variable', **kwargs):
        """
        Generate a KDE (Kernel Density Estimation) plot for multiple fits.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data for the KDE plot.
            parameters (list): List of parameter names to include in the KDE plot.
            hue (str, optional): Variable name for coloring the plot by fit. Default is 'fit'.
            **kwargs: Additional keyword arguments to pass to the informative_kde function.

        Returns:
            kdegrid: The Seaborn FacetGrid object containing the KDE plot.
        """
        def informative_kde(x=None, percs=[5, 50, 95], color='purple', median_color='black', label=None, bw_adjust=0.4):
            ax = sns.kdeplot(x, fill=False, bw_adjust=bw_adjust, color=color, label=label)
            kdeline = ax.lines[-1]
            x_data = kdeline.get_xdata()
            y_data = kdeline.get_ydata()
            left, middle, right = np.nanpercentile(x, percs)
            ax.vlines(middle, 0, np.interp(middle, x_data, y_data),
                      color=median_color, ls=':', linewidth=1.5, label='median')
            ax.fill_between(x_data, 0, y_data,
                            color=color, alpha=0.4, label=str(percs[0])+"-"+str(percs[2])+'%')
            ax.fill_between(x_data, 0, y_data, where=(left <=x_data) & (x_data <= right), interpolate=False, color=color, alpha=0.5, lw=.2)
            return ax

        dmelt = df.melt(id_vars=['fit'])
        kdegrid = sns.FacetGrid(dmelt, col=col, hue=hue, col_wrap=min(len(parameters), self.col_wrap), sharey=False, sharex=False, height=self.fig_scale)
        kdegrid.map(informative_kde, 'value', **kwargs)
        kdegrid.add_legend(label_order= kdegrid.hue_names + ['median'], title='',bbox_to_anchor=(1.2, 0.5))
        kdegrid.set_titles("")
        for i, ax in enumerate(kdegrid.axes.flatten()):
            ax.set_xlabel(parameters[i])
        return kdegrid

    @multi_fit_plot
    def ridgeplot(self, df, parameters, row='fit', col='variable', hue='fit', pcolor=-3, height=1):
        """
        Generate a ridge plot for multiple fits.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data for the ridge plot.
            parameters (list): List of parameter names to include in the ridge plot.
            row (str, optional): Variable name for organizing plots into rows. Default is 'fit'.
            col (str, optional): Variable name for organizing plots into columns. Default is 'variable'.
            hue (str, optional): Variable name for coloring the plot by fit. Default is 'fit'.
            pcolor (int, optional): The starting color value for the palette. Default is -3.

        Returns:
            grid: The Seaborn FacetGrid object containing the ridge plot.
        """
        n_fits = df.fit.nunique()
        def mykde(x=None, color='purple', label=None):
            ax = sns.kdeplot(x, clip_on=False, fill=False, alpha=1, linewidth=1.2, color="w")
            kdeline = ax.lines[0]
            x_data = kdeline.get_xdata()
            y_data = kdeline.get_ydata()
            median = np.nanpercentile(x, [50])
            ax.vlines(median, 0, np.interp(median, y_data,y_data), color="orange", ls=':')
            ax.fill_between(x_data, 0, y_data, color=color, alpha=1)

        with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
            dmelt = df.melt(id_vars=['fit'])
            pal = sns.cubehelix_palette(n_fits+3, rot=-.4, start=pcolor, reverse=True)
            grid = sns.FacetGrid(dmelt, row=row, hue=hue, col=col, aspect=self.fig_scale, height=height, palette=pal, sharex=True, sharey=False)

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
        return grid
