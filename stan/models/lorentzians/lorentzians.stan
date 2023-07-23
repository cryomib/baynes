functions{
  #include lorentzians_functions.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  
  int N_peaks;
  ordered[N_peaks] p_E0;
  vector<lower=0>[N_peaks] p_gamma;
  vector<lower=0>[N_peaks] p_i;
  real<lower=0> p_FWHM;
  int<lower=0> N_ev;
  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = p_FWHM / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 5 / dx)) * 2 + 1;

  vector[N_window] x_window = dx * get_centered_window(N_window);
  vector[N_bins + N_window] x_extended = extend_vector(x, dx, N_window);
}

parameters {
  ordered[N_peaks] E0;
  vector<lower=0>[N_peaks] gamma;
  vector<lower=0>[N_peaks] i;
  real<lower=0, upper=1> bkg;
  real<lower=0.1, upper=15> sigma;
}

model {
  E0 ~ normal(p_E0, p_sigma);
  gamma ~ normal(p_gamma, p_gamma/10);
  i ~ normal(p_i, p_i/5);
  bkg ~ beta(2.5, 40);

  sigma ~ normal(p_sigma, 2);
  
  if (prior == 0) {
    counts ~ poisson(spectrum(x_extended, x_window, sigma, bkg, E0, gamma, i) * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep; 
  counts_rep = poisson_rng(spectrum(x_extended, x_window, sigma, bkg, E0, gamma, i) * N_ev);

}

