#include "Ho_fit_functions.stan"

data {
  int<lower=1> N_bins;
  int<lower=1> N_window;
  vector[N_bins+1] x;
  array[N_bins] int counts;

  real<lower=0> N_ev;  
  real<lower=0> FWHM_sim;
  real<lower=0> Q_sim;
  real bkg_sim;
  real m_nu_sim;
}

transformed data {
  real sigma_sim = FWHM_sim / (2*sqrt(2*log(2)));
  int W = num_elements(x) + N_window - 1;
  real dx = abs(x[2] - x[1]);
  vector[N_window] window_x = linspaced_vector(N_window, 0, N_window - 1);
  window_x = window_x - N_window %/% 2;
  vector[W] full_x = linspaced_vector(W, 0, dx * (W - 1));
  full_x = full_x + min(x) - dx * (N_window %/% 2);

  int N_peaks = 6;
  vector[N_peaks] x0 = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] g = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] h = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  h = h/sum(h);
}

parameters {
  real<lower=0, upper=1> m_nu_red;
  real<lower=0> Q;  
  real<lower=0, upper=1> bkg;
  real<lower=0.1, upper=15> sigma;  

}

transformed parameters{
  real<lower=0, upper=150> m_nu = m_nu_red * 120;
}

model {
  m_nu_red ~ beta(1, 1.05);
  Q ~ normal(2838, 10);
  bkg ~ beta(1.5, 20);
  sigma ~ normal(5, 0.3);
  
  counts ~ poisson(spectrum(full_x, window_x, sigma/dx, bkg, m_nu, Q, x0, g, h, N_peaks)* N_ev);
}

generated quantities {
  int<lower = 0, upper = 1> lt_m  = m_nu < m_nu_sim;
  int<lower = 0, upper = 1> lt_Q  = Q < Q_sim;
  int<lower = 0, upper = 1> lt_b  = bkg < bkg_sim;
  int<lower = 0, upper = 1> lt_s  = sigma < sigma_sim;

}