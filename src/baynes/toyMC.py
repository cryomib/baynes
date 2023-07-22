from scipy import signal
from scipy import stats
from scipy import integrate
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


class SpectraSampler:
    def __init__(self, funcs_args_rates=None, flat_bkg=0, f_pileup=0, FWHM=0, n_events=1e5, n_spectra=100, ROI=[2650, 2900], dE=1, integrate=True):

        self.n_events = n_events
        self.n_spectra = n_spectra
        self.flat_bkg = flat_bkg
        self.FWHM = FWHM
        self.f_pu = f_pileup
        self.integrate = integrate
        self._E_max = 2900
        self.update_bins(dE=dE, ROI=ROI, FWHM=FWHM)
        self.update_spectrum(funcs_args_rates)

    def update_bins(self, dE=None, ROI=None, FWHM=None):
        if ROI is not None:
            self.ROI = ROI
            if ROI[1]>self._E_max:
                self._E_max = ROI[1]
        if dE is not None:
            self.dE = dE
        if FWHM is not None:
            self.FWHM = FWHM
            self.sigma = FWHM/(2*np.sqrt(2*np.log(2)))
            self.n_window = int(np.ceil(self.sigma/self.dE*6))*2 + 1

        half_window = int(np.floor((self.n_window - 1)/2))

        self.conv_window = signal.windows.gaussian(self.n_window, self.sigma/self.dE)
        self.conv_window = self.conv_window/sum(self.conv_window)
        self.bin_edges = np.arange(0, self._E_max + self.dE*(half_window+1), self.dE)
        self.n_bins = len(self.bin_edges) - 1

        self.ROI_idx = [np.argmin(np.abs(np.array(self.bin_edges)-self.ROI[0])),
                        np.argmin(np.abs(np.array(self.bin_edges)-self.ROI[1]))]
        self.ext_ROI_idx = [self.ROI_idx[0] - half_window, 
                            self.ROI_idx[1] + half_window]

        self.ROI_bin_edges = self.bin_edges[self.ROI_idx[0]:self.ROI_idx[1]+1]
        self.ROI_bin_centers = (self.bin_edges[self.ROI_idx[0]+1:self.ROI_idx[1]+1] 
                                + self.bin_edges[self.ROI_idx[0]:self.ROI_idx[1]])/2

    def update_spectrum(self, funcs_args_rates=None):
        if funcs_args_rates is None:
            funcs_args_rates = self.funcs_args_rates
        else:
            self.funcs_args_rates = funcs_args_rates
        self.spectrum = {}
        edges = self.bin_edges
        n_bins = self.n_bins
        spectrum = np.zeros(n_bins)
        sum_rates = 0
        for func_name, items in self.funcs_args_rates.items():
            func, args, rate = items
            if self.integrate is True:
                partial_sp = [integrate.quad(
                    func, edges[i], edges[i+1], args=tuple(args))[0] for i in range(n_bins)]
                partial_sp = np.array(partial_sp)
            else:         
                partial_sp = func(edges, *args)
                partial_sp = (partial_sp[:-1]+partial_sp[1:])/2

            partial_sp = (1-self.f_pu) * rate * partial_sp / sum(partial_sp) * 3600 * 24 / self.dE
            self.spectrum[func_name] = partial_sp
            spectrum = spectrum +  partial_sp
            sum_rates += rate

        if self.f_pu != 0:
            pileup_sp = signal.convolve(spectrum, spectrum, 'full')
            pileup_sp = self.f_pu * pileup_sp  * sum_rates / sum(pileup_sp) * 3600 * 24 / self.dE
            self.spectrum['pileup'] = pileup_sp
            spectrum = spectrum + pileup_sp[:n_bins]

        if self.flat_bkg!=0:
            flat_bkg = self.flat_bkg 
            self.spectrum ['flat'] = flat_bkg * np.repeat(1, n_bins)
            spectrum = spectrum + flat_bkg

        self.full_spectrum = spectrum 
        self.update_pdf()
    
    def update_pdf(self):
        weights_in_ROI = []
        binned_pdf = np.zeros(self.ROI_idx[1]-self.ROI_idx[0])
        for key, partial_sp in self.spectrum.items():
            partial_pdf = partial_sp[self.ext_ROI_idx[0]:self.ext_ROI_idx[1]]
            weights_in_ROI.append(sum(partial_sp[self.ROI_idx[0]: self.ROI_idx[1]]))
            if self.FWHM != 0:
                partial_pdf = signal.convolve(partial_pdf, self.conv_window, 'valid')
            binned_pdf = binned_pdf + partial_pdf
        weights_in_ROI = np.array(weights_in_ROI)
        self.weights_in_ROI = weights_in_ROI / sum(weights_in_ROI) 
        self.pdf_norm = sum(binned_pdf)
        self.binned_pdf = binned_pdf / sum(binned_pdf)

    def sample(self, poissonian=True):
        pdf = self.binned_pdf
        if poissonian:
            samples=np.random.poisson(pdf*self.n_events, size=(self.n_spectra, len(pdf)))
        else:
            samples = []
            for i in range(self.n_spectra):
                samples.append(random.multinomial(self.n_events, pdf))
            samples =  np.array(samples)
        return samples
    
    def plot_events(self, events, ax=None, scale='log'):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.ROI_bin_centers, self.binned_pdf * self.n_events, label='spectrum')
        ax.plot(self.ROI_bin_centers, events, label='events')
        ax.set_yscale(scale)
        ax.set_xlabel('E [eV]')
        ax.set_ylabel('Counts')
        ax.legend()
        return ax

    def plot_pdf(self, ax=None, scale='log', label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.ROI_bin_centers, self.binned_pdf, label=label)
        ax.set_yscale(scale)
        ax.set_xlabel('E [eV]')
        ax.set_ylabel('Prob(E)')
        return ax     

    def plot_spectrum(self, ax=None, scale='log'):
        x = (self.bin_edges[1:] + self.bin_edges[:-1])/2
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_yscale(scale)
        ax.axvspan(*self.ROI, alpha=0.4, color='grey', label='ROI')
        ax.plot(x, self.full_spectrum, label='total', lw=1.5)
        for key, partial_sp in self.spectrum.items():
            ax.plot(x, partial_sp[:len(x)], label=key, ls='--', lw=0.7)
        ax.legend(bbox_to_anchor=(1.05, 0.6))
        ax.set_xlabel('E [eV]')
        ax.set_ylabel('counts [$eV^{-1}day^{-1}det^{-1}$]')
        return ax   
    
    def get_measure_time(self, n_det = 32):
        ev_in_ROI = sum(self.full_spectrum[self.ROI_idx[0]:self.ROI_idx[1]]) * n_det * self.dE
        n_days = self.n_events / ev_in_ROI 
        print('estimated time: ', n_days, ' days')

    def set_measure_time(self, n_days, n_det = 32):
        ev_in_ROI = self.pdf_norm * n_det * self.dE
        self.n_events = int(np.floor(n_days * ev_in_ROI))
        print('Number of events in ROI: ', self.n_events)