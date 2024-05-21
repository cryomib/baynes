functions{
  #include convolution_functions.stan
  #include lorentzians_functions.stan
  #include spectra.stan

  vector peak_spectrum(vector x, real E0, real DE, vector p_gamma, vector p_i, vector y_bkg, vector)
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  int bkg_pars;
  int N_peaks;
  real p_E0;
  real DE;
  real p_bkg;
  vector[N_peaks] p_asymm;
  vector<lower=0>[N_peaks] p_gamma;
  vector<lower=0>[N_peaks] p_i;
  vector<lower=0>[2] p_lambda;
  vector<lower=0>[2] p_A_exp;
  real<lower=0> p_FWHM;
  int<lower=0> N_ev;
  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = p_FWHM / (2 * sqrt(2 * log(2)));
  int N_window = to_int(N_bins%/%2) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
  real xmin = extended_x[1];
  real xmax = extended_x[N_ext];
}

parameters {
  real x0;
  //vector<lower=0>[N_peaks] gamma;
  //vector<lower=0, upper=1>[N_peaks] i;
  //vector<lower=-1, upper=1>[N_peaks] asymm;
  vector<lower=0>[bkg_pars] y_bkg;
  real<lower=0, upper=3*p_sigma> sigma;
  positive_ordered[2] lambda_red;
  positive_ordered[2] A_exp_red;
}

transformed parameters {
  real<lower=0, upper=3*p_FWHM> FWHM=sigma*(2 * sqrt(2 * log(2)));
  real E0 = x0 + p_E0;
  positive_ordered[2] lambda = lambda_red*0.001;
  positive_ordered[2] A_exp = A_exp_red*0.01;
}

model {
  x0 ~ normal(0, 5);
  //gamma ~ normal(p_gamma, p_gamma/5);
  //asymm ~ normal(p_asymm, 0.05);
  //i ~ normal(p_i, p_i/4);
  y_bkg ~ normal(p_bkg, sqrt(p_bkg));
  lambda_red~normal(p_lambda, sqrt(p_lambda));
  A_exp_red~normal(p_A_exp, sqrt(p_A_exp));
  sigma ~ normal(p_sigma, 2);

  if (prior == 0) {
    vector[2] E0_vec = to_vector([E0-DE, E0]);
    vector[N_ext] bare_spectrum = rep_vector(y_bkg[1], N_ext);
    if (bkg_pars == 2){
      bare_spectrum += (y_bkg[2]-y_bkg[1])/(xmax-xmin)*(extended_x - xmin);
    }
    for (i in 1:N_peaks){
      bare_spectrum += p_i[i]*lorentzian_asymm(extended_x, E0_vec[i], p_gamma[i], p_asymm[i])*N_ev;
    }
    vector[N_window] response = gauss_plus_double_exp(window_x, 0, sigma, lambda, A_exp);
    vector[N_bins] convolved = convolve_and_bin(bare_spectrum, response);
    counts ~ poisson(convolved*N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep;
  {
    vector[2] E0_vec = to_vector([E0-DE, E0]);
    vector[N_ext] bare_spectrum = rep_vector(y_bkg[1], N_ext);
    if (bkg_pars == 2){
      bare_spectrum += (y_bkg[2]-y_bkg[1])/(xmax-xmin)*(extended_x - xmin);
    }
    for (i in 1:N_peaks){
      bare_spectrum += p_i[i]*lorentzian_asymm(extended_x, E0_vec[i], p_gamma[i], p_asymm[i])*N_ev;
    }
    vector[N_window] response = gauss_plus_double_exp(window_x, 0, sigma, lambda, A_exp);
    vector[N_bins] convolved = convolve_and_bin(bare_spectrum, response);
    counts_rep = poisson_rng(convolved*N_ev);
  }
}
