functions {
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  #include spectra.stan
}
data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  real m_max;
  real<lower=0> E_min;
  real<lower=0> N_ev;
  real<lower=0> N_tot;
  real<lower=0> FWHM;
  real<lower=0, upper=100> N_bkg;
  real<lower=0, upper=100> lambda;
  real<lower=0, upper=100> A_exp;
  real<lower=0> Q;
  real<lower=0, upper=m_max> m_nu;
  real<lower=0> f_pu;
}
transformed data {
  real sigma = FWHM / (2 * sqrt(2 * log(2)));
  real dx = abs(x[2] - x[1]);
  int N_window = to_int(floor(sigma * 14 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_bins] centers_x = head(x, N_bins) + dx / 2;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
}
generated quantities {
  vector[N_ext] E_ext;
  vector[N_window] E_response = window_x;
  {E_ext=extended_x;}
  array[N_bins] int counts_rep;
  vector[N_ext] bare_spectrum = Re187_bare(extended_x);

  vector[N_ext] spectrum = bare_spectrum;
  for (i in 1 : N_ext) {
    if (extended_x[i] < Q - m_nu) {
      spectrum[i] *= (Q - extended_x[i])
                     * sqrt((Q - extended_x[i]) ^ 2 - m_nu ^ 2);
    } else {
      spectrum[i] = 0;
    }
  }
  vector[N_window] response = gauss_plus_single_exp(window_x, 0, sigma,
                                                    lambda * 1e-3,
                                                    A_exp * 1e-2);
  vector[N_bins] pu = Re187_pileup(centers_x, Q, E_min);
  vector[N_bins] convolved = convolve_and_bin(spectrum, response);
 // counts_rep = poisson_rng(((N_ev - N_bkg * N_bins - N_tot * sum(pu) * f_pu)
  //                          * convolved + N_tot * pu * f_pu + N_bkg));
                            }
