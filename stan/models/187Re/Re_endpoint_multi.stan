functions {
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  #include spectra.stan
}
data {
  int<lower=1> N_bins;
  int N_det;
  vector[N_bins + 1] x;
  array[N_det, N_bins] int counts;
  real m_max;
  array[N_det] real<lower=0> N_ev;
  array[N_det] real<lower=0> N_tot;
  real<lower=0> p_Q;
  real<lower=0> p_std_Q;
  array[N_det] real<lower=0> p_FWHM;
  array[N_det] real<lower=0> p_std_FWHM;
  array[N_det] real p_bkg_alpha;
  array[N_det] real p_bkg_beta;
  int<lower=0, upper=1> prior;
  int<lower=0, upper=1> exp_model;
}
transformed data {
  real p_sigma = (p_FWHM[1] + p_std_FWHM[1]) / (2 * sqrt(2 * log(2)));
  real dx = abs(x[2] - x[1]);
  int N_window = to_int(floor(p_sigma * 10 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_bins] centers_x = head(x, N_bins) + dx / 2;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
}
parameters {
  real<lower=0, upper=1> m_red;
  real z;
  //  real xz;
  array[N_det] real<lower=0, upper=100> N_bkg;
  array[N_det] real<lower=0> f_pu;
  real<lower=0, upper=100> lambda;
  real<lower=0, upper=100> A_exp;
  //vector[3] exc;
  array[N_det] real<lower=0.1, upper=p_FWHM[1] * 3 + p_std_FWHM[1]> FWHM;
}
transformed parameters {
  real<lower=0> Q = z + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  // real <lower=0> A = 0.1 * xz + 1;
}
model {
  //exc~std_normal();
  m_red ~ beta(1, 1);
  z ~ normal(0, p_std_Q);
  N_bkg ~ gamma(p_bkg_alpha, p_bkg_beta);
  f_pu ~ gamma(2, 0.5);
  //  FWHM ~ normal(1, 0.05);
  FWHM ~ normal(p_FWHM, p_std_FWHM);
  if (exp_model == 0) {
    lambda ~ normal(46, 6);
    A_exp ~ normal(57, 34);
  } else {
    lambda ~ normal(30, 5);
    A_exp ~ normal(10, 10);
  }
  if (prior == 0) {
    vector[N_ext] spectrum = Re187(extended_x, m_nu, Q);
    vector[N_ext] pu = Re187_pileup(extended_x, Q);
    for (i in 1 : N_det) {
      vector[N_ext] exp_sp = (N_ev[i] - N_bkg[i] * N_ext
                              - sum(pu) * N_tot[i] * f_pu[i] * 1e-4)
                             * spectrum + pu * N_tot[i] * f_pu[i] * 1e-4
                             + N_bkg[i];
      vector[N_window] response = gauss_plus_single_exp(window_x, 0,
                                                        FWHM[i]
                                                        / (2 * sqrt(2 * log(2))),
                                                        lambda * 1e-3,
                                                        A_exp * 1e-2);
      vector[N_bins] convolved = convolve_and_bin(exp_sp, response);
      counts[i] ~ poisson(convolved * N_ev[i]);
    }
    //counts ~ poisson(spectrum(extended_x, window_x, FWHM, f_bkg, f_pu, pileup, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}
generated quantities {
  array[N_bins] int counts_rep;
  vector[N_bins] log_lik;
  {
    vector[N_ext] spectrum = Re187(extended_x, m_nu, Q);
    vector[N_ext] pu = Re187_pileup(extended_x, Q);
    int i = 1;
    vector[N_ext] exp_sp = (N_ev[i] - N_bkg[i] * N_ext
                            - sum(pu) * N_tot[i] * f_pu[i] * 1e-4)
                           * spectrum + pu * N_tot[i] * f_pu[i] * 1e-4
                           + N_bkg[i];
    vector[N_window] response = gauss_plus_single_exp(window_x, 0,
                                                      FWHM[i]
                                                      / (2 * sqrt(2 * log(2))),
                                                      lambda * 1e-3,
                                                      A_exp * 1e-2);
    vector[N_bins] convolved = convolve_and_bin(exp_sp, response);
    counts_rep = poisson_rng(convolved * N_ev[i]);
    for (n in 1 : N_bins) {
      log_lik[n] = poisson_lpmf(counts[i][n] | convolved * N_ev[i]);
    }
    // counts_rep = poisson_rng(spectrum(extended_x, window_x, FWHM, f_bkg, f_pu, pileup, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}

