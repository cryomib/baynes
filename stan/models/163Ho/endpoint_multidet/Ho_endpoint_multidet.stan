functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_bins;
  int<lower=1> N_det;

  vector[N_bins + 1] x;
  int counts[N_det, N_bins];
  
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

  vector[N_window] x_window = dx * get_centered_window(N_window);
  vector[N_ext] x_extended = extend_vector(x, dx, N_window);
  vector[N_ext] bare_spectrum = Ho_lorentzians(x_extended);
}

parameters {
  real<lower=0, upper=1> m_nu_red;
  real<lower=0> Q[N_det];
  real<lower=0, upper=1> bkg[N_det];
 // real<lower=0.1, upper=15> sigma;
}

transformed parameters {
  real<lower=0, upper=150> m_nu = m_nu_red * 150;
}

model {
  m_nu_red ~ beta(1, 1.05);
 // sigma ~ normal(p_sigma, 0.3);

  for (i in 1:N_det){
    Q[i] ~ normal(p_Q, 10);
    bkg[i] ~ beta(2.5, 40);
    if (prior == 0) {
      counts[i] ~ poisson(spectrum(x_extended, x_window, p_sigma, bkg[i], m_nu, Q[i], bare_spectrum) * N_ev);
    }
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(x_extended, x_window, p_sigma, bkg[1], m_nu, Q[1], bare_spectrum) * N_ev);
}
