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
  real<lower=0> N_ev;
  real<lower=0> N_tot;
  real<lower=0> p_Q;
  real<lower=0> p_std_Q;
  real<lower=0> p_FWHM;
  real<lower=0> p_std_FWHM;
  real p_bkg_alpha, p_bkg_beta;
  int<lower=0, upper=1> prior;
  int<lower=0, upper=1> exp_model;
  //real p_lambda;
  int<lower=0, upper=1> mod;
  real p_A_exp;

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
  real<lower=0> f_pu_red;
  real<lower=0, upper=100> lambda;
  real<lower=0, upper=100> A_exp;
  //vector[3] exc;
  real<lower=0.1, upper=p_FWHM*3+p_std_FWHM> FWHM;
}

transformed parameters {
  real <lower=0> Q = z + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  real <lower=0> A = 0.1 * xz + 1;
  real <lower=0> f_pu = f_pu_red*1e-4;

  real sigma = FWHM / (2 * sqrt(2 * log(2)));
  //vector[N_ext] sigma = FWHM*sqrt(87.1+15.6*extended_x*1e-3+0.65*(extended_x*1e-3)^2);
  //vector[3] exc_pars = 0.005*exc;
}

model {
  //exc~std_normal();
  m_red ~ beta(1, 1);
  z ~ normal(0, p_std_Q);
  xz~std_normal();
  N_bkg ~ gamma(p_bkg_alpha, p_bkg_beta);
  f_pu_red ~ gamma(2, 0.5);
//  FWHM ~ normal(1, 0.05);
  FWHM ~ normal(p_FWHM, p_std_FWHM);
  if (exp_model == 0){
    lambda ~ normal(46, 6);
    A_exp ~ normal(57, 34);
    }
  else{
    lambda ~ normal(27, 9);
    A_exp ~ normal(p_A_exp, 10);
  }
  if (prior == 0) {
        vector[N_ext] spectrum;
    if (mod == 0) {
      spectrum = Re187(extended_x, m_nu, Q);
      } 
    else {
      spectrum = Re187_gal(extended_x, m_nu, Q, 0, 0);
    }
   # vector[N_ext] spectrum = Re187(extended_x, m_nu, Q);
    vector[N_bins] pu = Re187_pileup(centers_x, Q);
    vector[N_window] response = gauss_plus_single_exp(window_x, 0, sigma, lambda*1e-3, A_exp*1e-2);
    vector[N_bins] convolved = convolve_and_bin(spectrum, response);
    counts ~ poisson(((N_ev-N_bkg*N_ext-N_tot*sum(pu)*f_pu)*convolved + N_tot*pu*f_pu + N_bkg)*A);
    //counts ~ poisson(spectrum(extended_x, window_x, FWHM, f_bkg, f_pu, pileup, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  vector[N_bins] log_lik;
  {
    vector[N_ext] spectrum;
    if (mod == 0) {
      spectrum = Re187(extended_x, m_nu, Q);
      } 
    else {
      spectrum = Re187_gal(extended_x, m_nu, Q, 0, 0);
    }
    vector[N_bins] pu = Re187_pileup(centers_x, Q);
    vector[N_window] response = gauss_plus_single_exp(window_x, 0, sigma, lambda*1e-3, A_exp*1e-2);
    vector[N_bins] convolved = convolve_and_bin(spectrum, response);
    counts_rep = poisson_rng(((N_ev-N_bkg*N_ext-N_tot*sum(pu)*f_pu)*convolved + N_tot*pu*f_pu + N_bkg)*A);   
    for (n in 1:N_bins) {
      log_lik[n] = poisson_lpmf(counts[n] |((N_ev-N_bkg*N_ext-N_tot*sum(pu)*f_pu)*convolved + N_tot*pu*f_pu + N_bkg)*A);
  }
    // counts_rep = poisson_rng(spectrum(extended_x, window_x, FWHM, f_bkg, f_pu, pileup, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}
