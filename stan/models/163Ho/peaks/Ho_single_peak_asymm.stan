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
  real<lower=0, upper=1> a_z;
  real b;
}

transformed parameters {
  #include base_transformed_parameters.stan
  real<lower=-1, upper=1> alpha = 2*a_z - 1;

}

model {
  //m_nu ~ uniform(0, m_max);
 // alpha~normal(0, 0.3);
  b~normal(0, 13);
  a_z ~ beta(20,20);
  #include base_model.stan

  if (prior == 0) {

      vector[N_ext] bare_spectrum = linear + 
                                    lorentzian_xps(extended_x, E_M1, gamma_M1, alpha, b)*100;
      counts ~ poisson(spectrum(extended_x, window_x, FWHM, 0, 2833, bare_spectrum) * A * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  vector[N_bins] log_lik;
  {
      vector[N_ext] bare_spectrum = y0 + m  * (extended_x - xmin)+ 
                                    lorentzian_xps(extended_x, E_M1, gamma_M1, alpha, b)*100;
      vector[N_bins] exp_spectrum = spectrum(extended_x, window_x, FWHM, 0, 2833, bare_spectrum) * A * N_ev;                      
      counts_rep = poisson_rng(exp_spectrum);
      for (n in 1:N_bins) {
        log_lik[n] = poisson_lpmf(counts[n] | exp_spectrum[n]);
      }
  }
}

