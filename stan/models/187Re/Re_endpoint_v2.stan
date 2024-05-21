functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  #include spectra.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  real m_max;
  real p_f_pu;
  real<lower=0> N_ev;
  real<lower=0> p_Q;
  real<lower=0> p_std_Q;
  real<lower=0> p_FWHM;
  real<lower=0> p_std_FWHM;
  int<lower=0, upper=1> prior;
  int<lower=0, upper=1> exp_model;

}

transformed data {
 real p_sigma = (p_FWHM+p_std_FWHM) / (2 * sqrt(2 * log(2)));
  real dx = abs(x[2] - x[1]);
  int N_window = to_int(floor(p_sigma * 10 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_bins] centers_x = head(x, N_bins) + dx/2;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window); 
}

parameters {
  real<lower=0, upper=1> m_red;
  real z;
  real xz;
  real<lower=0, upper=100> N_bkg;
  real<lower=0, upper=1> f_pu;
  real<lower=0, upper=100> lambda;
  real<lower=0, upper=100> A_exp;
  //vector[3] exc;
  real<lower=0.1, upper=p_FWHM*3+p_std_FWHM> FWHM;
}

transformed parameters {
  real <lower=0> Q = z + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  real <lower=0> A = 0.1 * xz + 1;
  real sigma = FWHM / (2 * sqrt(2 * log(2)));
  //vector[N_ext] sigma = FWHM*sqrt(87.1+15.6*extended_x*1e-3+0.65*(extended_x*1e-3)^2);
  //vector[3] exc_pars = 0.005*exc;
}

model {
  //exc~std_normal();
  m_red ~ beta(1, 1);
  z ~ normal(0, p_std_Q);
  xz~std_normal();
  N_bkg ~ normal(0.25*dx, 0.5);
  f_pu ~ normal(p_f_pu, 3*p_f_pu);
//  FWHM ~ normal(1, 0.05);
  FWHM ~ normal(p_FWHM, p_std_FWHM);
  if (exp_model == 0){
    lambda ~ normal(46, 6);
    A_exp ~ normal(57, 34);
    }
  else{
    lambda ~ normal(30, 5);
    A_exp ~ normal(10, 10);
  }
  if (prior == 0) {
    vector[N_ext] spectrum = (1-f_pu)*Re187(extended_x, m_nu, Q);
    spectrum += f_pu * Re187_pileup(extended_x, Q);
    spectrum = (N_ev-N_bkg*N_ext)*spectrum + N_bkg;
    vector[N_window] response = gauss_plus_single_exp(window_x, 0, sigma, lambda*1e-3, A_exp*1e-2);
    vector[N_bins] convolved = convolve_and_bin(spectrum, response);
    counts ~ poisson(convolved*N_ev*A);
    //counts ~ poisson(spectrum(extended_x, window_x, FWHM, f_bkg, f_pu, pileup, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  vector[N_bins] log_lik;
  {

    vector[N_ext] spectrum = (1-f_pu)*Re187(extended_x, m_nu, Q);
    spectrum += f_pu * Re187_pileup(extended_x, Q);
    spectrum = (N_ev-N_bkg*N_ext)*spectrum + N_bkg;
    vector[N_window] response = gauss_plus_single_exp(window_x, 0, sigma, lambda*1e-3, A_exp*1e-2);
    vector[N_bins] convolved = convolve_and_bin(spectrum, response);
    counts_rep = poisson_rng(convolved*N_ev*A);   
    for (n in 1:N_bins) {
      log_lik[n] = poisson_lpmf(counts[n] | convolved*N_ev*A);
    }
    // counts_rep = poisson_rng(spectrum(extended_x, window_x, FWHM, f_bkg, f_pu, pileup, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}
