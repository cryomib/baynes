functions{
  #include convolution_functions.stan
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
  int<lower=0, upper=1> prior;
}

transformed data {
  int<lower=0> N_ev = sum(counts);
  real dx = abs(x[2] - x[1]);
  real p_sigma = p_FWHM / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 5 / dx)) * 2 + 1;

  real E_scale = (x[N_bins+1]-x[1])/2;
  real E_shift = (x[1]+x[N_bins+1])/2;
  vector[N_peaks] p_z0 = (p_E0-E_shift)/E_scale;
  vector[N_window] x_window = dx * get_centered_window(N_window);
  vector[N_bins + N_window] x_extended = extend_vector(x, dx, N_window);
  print(x_window);
  print(x_extended);
}

parameters {
  ordered[N_peaks] z0;
  vector<lower=0>[N_peaks] gamma;
  vector<lower=0>[N_peaks] i;
  real<lower=0, upper=20> FWHM;
}

transformed parameters {
  ordered[N_peaks] E0 = z0 * E_scale + E_shift;
}

model {
  z0 ~ normal(p_z0, 0.1);
  gamma ~ normal(p_gamma, p_gamma/6);
  i ~ normal(p_i, p_i/5);
  FWHM ~ normal(p_FWHM, 3);

  if (prior == 0) {
    counts ~ poisson(spectrum(x_extended, x_window, FWHM, 0, E0, gamma, i) * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  counts_rep = poisson_rng(spectrum(x_extended, x_window, FWHM, 0, E0, gamma, i) * N_ev);
}