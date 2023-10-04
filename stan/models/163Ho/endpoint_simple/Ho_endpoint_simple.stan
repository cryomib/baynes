functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  real m_max;
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
  vector[N_ext] bare_spectrum = Ho_lorentzians(extended_x);
}

parameters {
  real<lower=0, upper=1> m_red;
  real z;
  real<lower=0, upper=1> f_bkg;
  real<lower=0.1, upper=15> FWHM;
}

transformed parameters {
  real <lower=0> Q = z + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;

}

model {
  //m_nu ~ uniform(0, m_max);
  m_red ~ beta(1, 1);
  z ~ normal(0, 33.54);
  f_bkg ~ beta(1.8, 30);
  FWHM ~ normal(p_FWHM, 1);
  if (prior == 0) {
    counts ~ poisson(spectrum(extended_x, window_x, FWHM, f_bkg, m_nu, Q, bare_spectrum) * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(extended_x, window_x, FWHM, f_bkg, m_nu, Q, bare_spectrum) * N_ev);
}

