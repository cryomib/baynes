/*

IMPORTANT: INCREASE NUMERICAL PRECISION IN CMDSTANPY TO  MAKE THIS WORK PROPERLY
USE SIG_FIGS = 10 IN SAMPLER ARGUMENTS

*/


functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  #include lorentzians_functions.stan
  real partial_sum
}

data {
  int fit_peak;
  int<lower=1> N_bins;
  int<lower=1> N_det;
  vector[N_bins + 1] x;
  int counts[N_det, N_bins];
  real<lower=0> N_ev[N_det];
  vector[N_det] NEP;

  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = 15 / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 4 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);

  int N_peaks = 5;
  vector[N_peaks] E_H = tail(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = tail(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = tail(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  real p_E0 = 2047;
  real xmin = extended_x[1];
  real xmax = extended_x[N_ext];
}

parameters {
  real E_z;
  real<lower=0> gamma_M1;
  real E_sigma;
  real E_syst[N_det];
  real<lower=0, upper=1> a_z;
  real<lower=0.1, upper=20> FWHM[N_det];
  real<lower=0> A[N_det];
  real<lower = -1> y0;
  real<lower = -1> y1;

}

transformed parameters {
  real <lower=0> E_M1 = 50 * E_z + p_E0;
  real<lower=-1, upper=1> alpha = 2*a_z - 1;
  real m = (y1-y0)/(xmax-xmin);
}

model {
  //m_nu ~ uniform(0, m_max);
  E_sigma ~ gamma(8, 1);
  E_syst ~ normal(0, E_sigma);
  a_z ~ beta(20,20);
  y0 ~ normal(0, 1);
  y1 ~ normal(0, 1);

  A ~ normal(1, 0.05);
  E_z ~ std_normal();
  gamma_M1 ~ gamma(6.6, 0.5);
  FWHM ~ frechet(2*NEP, 0.5+NEP);
  for (i in 1:N_det){

    if (prior == 0) {
      vector[N_ext] bare_spectrum = y0 + m * (extended_x -E_syst[i]- xmin)+ 
                                    lorentzian_xps(extended_x-E_syst[i], E_M1, gamma_M1, alpha, 0) * 100;
      counts[i] ~ poisson(spectrum(extended_x-E_syst[i], window_x, FWHM[i], 0, 2833, bare_spectrum) *A[i]* N_ev[i]);
    }
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  {
      vector[N_ext] bare_spectrum = y0 + m * (extended_x -E_syst[1]- xmin) + 
                                    lorentzian_xps(extended_x-E_syst[1], E_M1, gamma_M1, alpha, 0) * 100;
      vector[N_bins] exp_spectrum = spectrum(extended_x-E_syst[1], window_x, FWHM[1], 0, 2833, bare_spectrum)*A[1] * N_ev[1];   
      counts_rep = poisson_rng(exp_spectrum);                   
  }
}

