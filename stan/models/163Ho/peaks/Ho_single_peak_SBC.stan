/*

IMPORTANT: INCREASE NUMERICAL PRECISION IN CMDSTANPY TO  MAKE THIS WORK PROPERLY
USE SIG_FIGS = 10 IN SAMPLER ARGUMENTS

*/


functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
 // #include lorentzians_functions.stan
}

data {
  int fit_peak;
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  real<lower=0> N_ev;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = 15 / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 5 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);

  int N_peaks = 6;
  vector[N_peaks] E_H = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  real p_E0 = 2047;

  real<lower=0> gamma_M1_sim = gamma_rng(6.3, 0.5);
  real<lower=0.1, upper=20> FWHM_sim = gamma_rng(8,1);
  real <lower=0> E_M1_sim = normal_rng(2047, 30);
  real <lower=0> A_sim = normal_rng(1,0.05);
  vector[N_ext] bare_spectrum_sim;
  {
    vector[N_peaks] E0s = E_H;
    vector[N_peaks] gammas = gamma_H;
    E0s[fit_peak] = E_M1_sim;
    gammas[fit_peak] = gamma_M1_sim;
    bare_spectrum_sim = Ho_lorentzians(extended_x, E0s, gammas, i_H);
  }
  int counts[N_bins] = poisson_rng(spectrum(extended_x, window_x, FWHM_sim, 0, 2833, bare_spectrum_sim)*A_sim* N_ev);   
}
parameters {
  real E_z;
  real z;
  real<lower=0> gamma_M1;
  real<lower=0.1, upper=20> FWHM;
}

transformed parameters {
  real <lower=0> E_M1 = 30 * E_z + p_E0;
  real <lower=0> A = 0.05 * z + 1;
}

model {
  //m_nu ~ uniform(0, m_max);
  E_z ~ std_normal();
  z ~ std_normal();
  gamma_M1 ~ gamma(6.3, 0.5);
  FWHM ~ gamma(8, 1);
      vector[N_peaks] E0s = E_H;
      vector[N_peaks] gammas = gamma_H;
  //    vector[N_peaks] is = i_H;
      E0s[fit_peak] = E_M1;
      gammas[fit_peak] = gamma_M1;
      //is[fit_peak] = I_M1;
      vector[N_ext] bare_spectrum = Ho_lorentzians(extended_x, E0s, gammas, i_H);
      counts ~ poisson(spectrum(extended_x, window_x, FWHM, 0, 2833, bare_spectrum)*A* N_ev);
  
}

generated quantities {
  int<lower = 0, upper = 1> lt_E_M1  = E_M1 < E_M1_sim;
  int<lower = 0, upper = 1> lt_gamma_M1  = gamma_M1 < gamma_M1_sim;
  int<lower = 0, upper = 1> lt_FWHM  = FWHM < FWHM_sim;
  int<lower = 0, upper = 1> lt_A  = A < A_sim;

}

