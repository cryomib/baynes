#include Ho_fit_functions.stan

data {
  int<lower=1> N_bins;
  int<lower=1> N_window;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  
  real<lower=0> N_ev;
  real<lower=0> p_Q;
  real<lower=0> p_FWHM;
  int<lower=0, upper=1> prior;
}

transformed data {
  real p_sigma = p_FWHM / (2 * sqrt(2 * log(2)));
  int W = num_elements(x) + N_window - 1;
  real dx = abs(x[2] - x[1]);
  vector[N_window] window_x = linspaced_vector(N_window, 0, N_window - 1);
  window_x = window_x - N_window %/% 2;
  vector[W] full_x = linspaced_vector(W, 0, dx * (W - 1));
  full_x = full_x + min(x) - dx * (N_window %/% 2);
  
  int N_peaks = 6;
  vector[N_peaks] E_H = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  i_H = i_H/sum(i_H);

  real max_x = max(full_x);
  int N_tot = to_int(floor(max_x/dx))+1;
  vector[N_tot] x_pu = linspaced_vector(N_tot, 0, (N_tot-1))*dx;
  print(x_pu);
  vector[N_tot] Ho_edges = Ho_first_order(x_pu, 0, p_Q, E_H, gamma_H, i_H);
  vector[N_tot-1] Ho_centers = (head(Ho_edges, N_tot-1) + tail(Ho_edges, N_tot-1))/2;;

  vector[W-1] pu_spectrum = segment(autocorrelation(Ho_centers), N_tot-W+1, W-1);

}

parameters {
  real<lower=0, upper=1> m_nu_red;
  real<lower=0> Q;
  real<lower=0, upper=1> bkg;
  real<lower=0, upper=1> pu;

  real<lower=0.1, upper=15> sigma;
}

transformed parameters {
  real<lower=0, upper=150> m_nu = m_nu_red * 120;
}

model {
  m_nu_red ~ beta(1, 1.05);
  Q ~ normal(p_Q, 10);
  bkg ~ beta(2.5, 40);
  pu ~ beta(2.5, 10);

  sigma ~ normal(p_sigma, 0.3);
  
  if (prior == 0) {
    counts ~ poisson(spectrum(full_x, window_x, sigma / dx, bkg, pu, pu_spectrum, m_nu, Q, E_H,
                              gamma_H, i_H, N_peaks)
                              * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(full_x, window_x,
                                                      sigma / dx, bkg, pu, pu_spectrum, m_nu,
                                                      Q, E_H, gamma_H, i_H, N_peaks)
                                                      * N_ev);
}

