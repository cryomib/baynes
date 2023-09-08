functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_ev;
  vector[N_ev] events;
  array[N_bins] int counts;
  
  real<lower=0> p_Q;
  real<lower=0> p_FWHM;
  int<lower=0, upper=1> prior;
}

transformed data {
  int N_peaks = 6;
  vector[N_peaks] E_H = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
}

parameters {
  real<lower=0, upper=1> m_nu_red;
  real<lower=0> Q;
  real<lower=0, upper=1> bkg;
  real<lower=0.1, upper=15> sigma;
  vector<lower=0> true_evs;
}

transformed parameters {
  real<lower=0, upper=150> m_nu = m_nu_red * 150;
}

model {
  m_nu_red ~ beta(1, 1.05);
  Q ~ normal(p_Q, 10);
  bkg ~ beta(2.5, 40);
  sigma ~ normal(p_sigma, 0.3);
  events ~ normal(true_evs, sigma)
  if (prior == 0) {
    counts ~ poisson(Ho_first_order);
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(extended_x, window_x, sigma, bkg, m_nu, Q, bare_spectrum) * N_ev);
}

