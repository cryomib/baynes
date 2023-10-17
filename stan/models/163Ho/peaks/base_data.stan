  int fit_peak;
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  real<lower=0> N_ev;
  real<lower=0> p_FWHM;
  int<lower=0, upper=1> prior;