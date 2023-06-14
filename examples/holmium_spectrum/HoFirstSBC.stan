#include "Ho_fit_functions.stan"

data {
  int<lower=1> N_bins;
  int<lower=1> N_window;
  int N_counts;
  vector[N_bins+1] x;
  // priors parameters
  real<lower=0> p_Q;
  real<lower=0> p_m;
  real<lower=0> p_sigma;


}

transformed data {
  int W = num_elements(x) + N_window - 1;
  real dx = abs(x[2] - x[1]);
  vector[N_window] window_x = linspaced_vector(N_window, 0, dx * (N_window - 1));
  window_x = window_x - dx * N_window %/% 2;
  vector[W] full_x = linspaced_vector(W, 0, dx * (W - 1));
  full_x = full_x + min(x) - dx * (N_window %/% 2);

  int N_peaks = 2;
  vector[N_peaks] x0 = to_vector([2047, 1842]);//, 414.2, 333.5, 49.9, 26.3]);
  vector[N_peaks] g = to_vector([13.2, 6.0]);//, 5.4, 5.3, 3.0, 3.0]);
  vector[N_peaks] h = to_vector([1, 0.0526]);//, 0.2329, 0.0119, 0.0345, 0.0015]);
  h = h/sum(h);

  real<lower=0> m_nu_sim = exponential_rng(p_m);
  real<lower=0> Q_sim = normal_rng(p_Q, 5);
  real<lower=0, upper=1> bkg_sim = beta_rng(0.5, 4);  
  real<lower=0> sigma_sim = chi_square_rng(p_sigma);

  print(m_nu_sim);
  print(Q_sim);
  print(bkg_sim);
  array[N_bins] int counts = multinomial_rng(spectrum(full_x, window_x, sigma_sim, bkg_sim, 
                                                 m_nu_sim, Q_sim, x0, g, h, N_peaks), N_counts);
}

parameters {
  real<lower=0> m_nu;
  real<lower=0> Q;  
  real<lower=0, upper=1> bkg;
  real<lower=0> sigma;  
}

model {
  m_nu ~ exponential(p_m);
  Q ~ normal(p_Q, 5);
  bkg ~ beta(0.5, 4);
  sigma ~ chi_square(p_sigma);
  counts ~ poisson(spectrum(full_x, window_x, sigma, bkg, m_nu, Q, x0, g, h, N_peaks)*N_counts); 
}

generated quantities {
  int<lower = 0, upper = 1> lt_m  = m_nu < m_nu_sim;
  int<lower = 0, upper = 1> lt_Q  = Q < Q_sim;
  int<lower = 0, upper = 1> lt_b  = bkg < bkg_sim;
  int<lower = 0, upper = 1> lt_s  = sigma < sigma_sim;
  //array[N_bins] int y_rep = poisson_rng(spectrum(full_x, window_x, sigma, bkg, m_nu, Q, x0, g, h, N_peaks)*N_counts);


}