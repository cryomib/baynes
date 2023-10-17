/*

IMPORTANT: INCREASE NUMERICAL PRECISION IN CMDSTANPY TO  MAKE THIS WORK PROPERLY
USE SIG_FIGS = 10 IN SAMPLER ARGUMENTS

*/


functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  #include lorentzians_functions.stan
}

data {
 #include base_data.stan
}

transformed data {
 #include base_transformed_data.stan
}

parameters {
 #include base_parameters.stan
}

transformed parameters {
 #include base_transformed_parameters.stan
}

model {
  #include base_model.stan

  if (prior == 0) {
      vector[N_ext] bare_spectrum = lorentzian(extended_x, E_M1, gamma_M1)*100 + linear;
      counts ~ poisson(spectrum(extended_x, window_x, FWHM, 0, 2833, bare_spectrum)* N_ev *A);
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  vector[N_bins] log_lik;
  {
      vector[N_ext] bare_spectrum = lorentzian(extended_x, E_M1, gamma_M1) * 100 + y0 + m * (extended_x - xmin);
      vector[N_bins] exp_spectrum = spectrum(extended_x, window_x, FWHM, 0, 2833, bare_spectrum)* N_ev *A;                      
      counts_rep = poisson_rng(exp_spectrum);
      for (n in 1:N_bins) {
        log_lik[n] = poisson_lpmf(counts[n] | exp_spectrum[n]);
      }
  }
}
