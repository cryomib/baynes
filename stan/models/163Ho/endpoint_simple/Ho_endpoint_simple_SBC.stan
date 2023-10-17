functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  real m_max;
  real<lower=0> N_ev;
  real<lower=0> p_Q;
  real<lower=0> p_FWHM;
}

transformed data {

  real p_sigma = (p_FWHM) / (2 * sqrt(2 * log(2)));
  #include Ho_transformed_data.stan

  vector[N_ext] bare_spectrum = Ho_lorentzians(extended_x);
  real<lower=0, upper=m_max> m_nu_sim = beta_rng(1, 1)*m_max;
  real<lower=0> Q_sim = normal_rng(p_Q, 33.54);
  real<lower=0, upper=1> f_bkg_sim = beta_rng(1.8, 30);
  real<lower=0.1, upper=15> FWHM_sim = normal_rng(p_FWHM, 1); 
  array[N_bins] int counts = poisson_rng(spectrum(extended_x-m_nu_sim+50, window_x, FWHM_sim, f_bkg_sim, m_nu_sim, Q_sim, bare_spectrum) * N_ev);

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
  m_red ~ beta(1, 1);
  z ~ normal(0, 33.54);
  f_bkg ~ beta(1.8, 30);
  FWHM ~ normal(p_FWHM, 1);
  counts ~ poisson(spectrum(extended_x-m_nu_sim+50, window_x, FWHM, f_bkg, m_nu, Q, bare_spectrum) * N_ev);
}

generated quantities {
  int<lower = 0, upper = 1> lt_m_nu  = m_nu < m_nu_sim;
  int<lower = 0, upper = 1> lt_Q  = Q < Q_sim;
  int<lower = 0, upper = 1> lt_f_bkg  = f_bkg < f_bkg_sim;
  int<lower = 0, upper = 1> lt_FWHM  = FWHM < FWHM_sim;
}

