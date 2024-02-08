/*

IMPORTANT: INCREASE NUMERICAL PRECISION IN CMDSTANPY TO  MAKE THIS WORK PROPERLY
USE "sig_figs" = 9 OR HIGHER IN SAMPLER ARGUMENTS

*/


functions{
  #include convolution_functions.stan
  #include lorentzians_functions.stan
}

data {
  int fit_peak;
  int<lower=1> N_bins;
  int<lower=1> N_det;
  vector[N_bins + 1] x;
  int counts[N_det, N_bins];
  real<lower=0> N_ev[N_det];
  vector[N_det] NEP;
  real p_E0;
  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = (NEP[1] + 4) / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 4 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
  real xmin = extended_x[1];
  real xmax = extended_x[N_ext];
}

parameters {
  real E_z;
  real<lower=0> gamma;
  real E_sigma;
  real E_syst[N_det];
  real<lower=0, upper=1> a_z;
  real<lower=0.1, upper=20> FWHM[N_det];
  real<lower=0> A[N_det];
  real<lower = 0> y0;
  real<lower = 0> y1;

}

transformed parameters {
  real <lower=0> E0 = 50 * E_z + p_E0;
  real<lower=-1, upper=1> alpha = 2*a_z - 1;
  real m = (y1-y0)/(xmax-xmin);
}

model {
  E_sigma ~ gamma(8, 1); // hierarchical prior
  E_syst ~ normal(0, E_sigma);
  a_z ~ beta(20,20);
  y0 ~ normal(0, 1);
  y1 ~ normal(0, 1);

  A ~ normal(1, 0.05);
  E_z ~ std_normal();
  gamma ~ gamma(6.6, 0.5);
  FWHM ~ frechet(2*NEP, 0.5+NEP); // prior  between NEP - 0.5 eV and NEP + 4 eV
  for (i in 1:N_det){

    if (prior == 0) {
      vector[N_ext] bare_spectrum = y0 + m * (extended_x -E_syst[i]- xmin)+ 
                                    lorentzian_xps(extended_x-E_syst[i], E0, gamma, alpha, 0) * 100; // factor 100 to consider low background (y0, y1)
      vector[N_bins] convolved = convolve_and_bin(window_x, bare_spectrum, FWHM[i]);
      counts[i] ~ poisson(convolved * A[i] * N_ev[i]);
    }
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  {
      vector[N_ext] bare_spectrum = y0 + m * (extended_x -E_syst[1]- xmin) + 
                                    lorentzian_xps(extended_x-E_syst[1], E0, gamma, alpha, 0) * 100;
      vector[N_bins] convolved = convolve_and_bin(window_x, bare_spectrum, FWHM[1]);
      counts_rep = poisson_rng(convolved * A[1] * N_ev[1]);                   
  }
}

