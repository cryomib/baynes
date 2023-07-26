functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  
  real<lower=0> N_ev;
  real<lower=0> p_Q;
  real<lower=0> p_FWHM;
  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = p_FWHM / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 5 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;

  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);

  int N_peaks = 6;
  vector[N_peaks] E_H = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  vector[N_ext-1] pu_spectrum = Ho_pileup(extended_x, dx, 0, p_Q, E_H, gamma_H, i_H);
}

parameters {
  real<lower=0, upper=1> m_nu_red;
  real<lower=0> Q;
  real<lower=0, upper=1> bkg;
  real<lower=0, upper=1> pu;
  real<lower=0.1, upper=15> sigma;
}

transformed parameters {
  real<lower=0, upper=150> m_nu = m_nu_red * 150;
}

model {
  m_nu_red ~ beta(1, 1.05);
  Q ~ normal(p_Q, 10);
  bkg ~ beta(2.5, 40);
  pu ~ beta(2.5, 20);
  sigma ~ normal(p_sigma, 0.3);
  if (prior == 0) {
    counts ~ poisson(spectrum(extended_x, window_x, sigma, bkg, pu, pu_spectrum, m_nu, Q, E_H, gamma_H, i_H) * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(extended_x, window_x, sigma, bkg, pu, pu_spectrum, m_nu, Q, E_H, gamma_H, i_H) * N_ev);
}

