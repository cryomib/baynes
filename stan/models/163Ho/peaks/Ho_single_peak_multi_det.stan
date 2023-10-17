/*

IMPORTANT: INCREASE NUMERICAL PRECISION IN CMDSTANPY TO  MAKE THIS WORK PROPERLY
USE SIG_FIGS = 10 IN SAMPLER ARGUMENTS

*/


functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
}

data {
  int fit_peak;
  int<lower=1> N_bins;
  int<lower=1> N_det;
  vector[N_bins + 1] x;
  int counts[N_det, N_bins];
  real<lower=0> N_ev[N_det];
  int<lower=0, upper=1> prior;
}

transformed data {
  real p_sigma = (15) / (2 * sqrt(2 * log(2)));
  #include Ho_transformed_data.stan
  real p_E0 = E_H[fit_peak];
  real p_E0 = E_H[fit_peak];
}

parameters {
  real E_z;
  real<lower=0> gamma;
  real E_sigma;
  real E_syst[N_det];
  real<lower=0.1, upper=20> FWHM[N_det];
}

transformed parameters {
  real <lower=0> E_0 = 50 * E_z + p_E0;
}

model {
  //m_nu ~ uniform(0, m_max);
  E_sigma ~ gamma(8, 1);
  E_syst ~ normal(0, E_sigma);
 
  E_z ~ std_normal();
  gamma ~ gamma(6, 0.5);

  vector[N_peaks] E0s = E_H;
  vector[N_peaks] gammas = gamma_H;
  E0s[fit_peak] = E_0;
  gammas[fit_peak] = gamma;

  for (i in 1:N_det){
    FWHM[i] ~ gamma(8, 1);
    if (prior == 0) {
      vector[N_ext] bare_spectrum = Ho_lorentzians(extended_x-E_syst[i], E0s, gammas, i_H);
      counts[i] ~ poisson(spectrum(extended_x-E_syst[i], window_x, FWHM[i], 0, 2833, bare_spectrum) * N_ev[i]);
    }
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  {   
      int i=1;
      vector[N_peaks] E0s = E_H;
      vector[N_peaks] gammas = gamma_H;
      E0s[fit_peak] = E_0;
      gammas[fit_peak] = gamma;
      vector[N_ext] bare_spectrum = Ho_lorentzians(extended_x-E_syst[i], E0s, gammas, i_H);
      counts_rep = poisson_rng(spectrum(extended_x-E_syst[i], window_x, FWHM[i], 0, 2833, bare_spectrum) * N_ev[i]);
  }
}

