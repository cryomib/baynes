functions {
  #include convolution_functions.stan
  #include Ho_fit_functions.stan

  #include spectra.stan
}
data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  real m_max;
  real<lower=0> E_min;
  real<lower=0> N_ev;
  real<lower=0> N_tot;
  real<lower=0> p_Q;
  real<lower=0> p_std_Q;
  real<lower=0> FWHM;
  real<lower=0> lambda;
  real<lower=0> A_exp;
  real p_bkg_alpha, p_bkg_beta;
  int<lower=0, upper=1> prior;
}
transformed data {
  real sigma = (FWHM) / (2 * sqrt(2 * log(2)));
  real dx = abs(x[2] - x[1]);
  int N_window = to_int(floor(sigma * 8 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_bins] centers_x = head(x, N_bins) + dx / 2;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
  vector[N_ext] bare_spectrum = Re187_bare(extended_x);
}
parameters {
  real<lower=0, upper=1> m_red;
  real z;
  real xz;
  real<lower=0, upper=100> N_bkg;
  real<lower=0> f_pu_red;
}
transformed parameters {
  real<lower=0> Q = z * p_std_Q + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  real<lower=0> A = 1 / sqrt(N_ev) * xz + 1;
  real<lower=0> f_pu = f_pu_red * 1e-4;
}

model {
  m_red ~ beta(1, 1);
  z ~ std_normal();
  xz ~ std_normal();
  N_bkg ~ gamma(p_bkg_alpha, p_bkg_beta);
  f_pu_red ~ gamma(2, 0.5);

  if (prior == 0) {
    vector[N_ext] spectrum = bare_spectrum;
    for (i in 1 : N_ext) {
      if (extended_x[i] < Q - m_nu) {
        spectrum[i] *= (Q - extended_x[i])
                       * sqrt((Q - extended_x[i]) ^ 2 - m_nu ^ 2);
      } else {
        spectrum[i] = 0;
      }
    }
    vector[N_bins] pu = Re187_pileup(centers_x, Q, E_min);
    vector[N_window] response = gauss_plus_single_exp(window_x, 0, sigma,
                                                      lambda * 1e-3,
                                                      A_exp * 1e-2);
    vector[N_bins] convolved = convolve_and_bin(spectrum, response);
    counts ~ poisson(((N_ev - N_bkg * N_bins - N_tot * sum(pu) * f_pu)
                      * convolved + N_tot * pu * f_pu + N_bkg)
                     * A);
  }
}
generated quantities {
  array[N_bins] int counts_rep;
  //vector[N_bins] log_lik;

  {
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
    counts_rep = poisson_rng(((N_ev - N_bkg * N_bins - N_tot * sum(pu) * f_pu)
                              * convolved + N_tot * pu * f_pu + N_bkg)
                             * A);
  }
}
