#include "Ho_fit_functions.stan"

data {
  int<lower=1> N_bins;
  int<lower=1> N_window;
  int<lower=1> N_peaks;
  vector[N_bins+1] x;
  array[N_bins] int counts;

  real<lower=0> p_Q;
  real<lower=0> p_m;
  real<lower=0> p_FWHM;
  int<lower=0, upper=1> prior;
  real<lower=0> p_N;  

}

transformed data {
  real p_sigma = p_FWHM / (2*sqrt(2*log(2)));
  int N_counts = sum(counts);
  int W = num_elements(x) + N_window - 1;
  real dx = abs(x[2] - x[1]);
  vector[N_window] window_x = linspaced_vector(N_window, 0, N_window - 1);
  window_x = window_x - N_window %/% 2;
  vector[W] full_x = linspaced_vector(W, 0, dx * (W - 1));
  full_x = full_x + min(x) - dx * (N_window %/% 2);

  vector[N_peaks] x0 = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] g = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] h = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  h = h/sum(h);
}

parameters {
  real<lower=0, upper=N_bins> m_nu;
  real<lower=0> Q;  
  real<lower=0, upper=1> bkg;
  real<lower=0.1, upper=15> sigma;  
  //real<lower=0> N_ev;  

}

model {
  m_nu ~ normal(0, p_m);
  Q ~ normal(p_Q, 10);
  bkg ~ beta(1, 2);
  sigma ~ normal(p_sigma, 0.5);
  #N_ev ~ normal(p_N, sqrt(p_N));
  #N_counts ~ poisson(N_ev);
  
  if (prior == 0) {
    counts ~ poisson(spectrum(full_x, window_x, sigma/dx, bkg, m_nu, Q, x0, g, h, N_peaks)*p_N);
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(full_x, window_x, sigma/dx, bkg, m_nu, Q, x0, g, h, N_peaks) * p_N);
}