functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  
  int N_peaks;
  ordered[N_peaks] p_E0;
  //vector<lower=0>[N_peaks] p_gamma;
  vector<lower=0>[N_peaks] p_i;
  real<lower=0> p_Q;
  real<lower=0> p_FWHM;
  real<lower=0> m_nu;
  int<lower=0> N_ev;
  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = p_FWHM / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 5 / dx)) * 2 + 1;

  vector[N_window] x_window = dx * get_centered_window(N_window);
  vector[N_bins + N_window] x_extended = extend_vector(x, dx, N_window);
  
  int N_fixed = 6 - N_peaks;
  vector[N_fixed] E_H = tail(to_vector([26.3, 49.9, 333.5, 414.2, 1842, 2047]), N_fixed);
  vector[N_fixed] gamma_H = tail(to_vector([3.0, 3.0, 5.3, 5.4, 6.0, 13.2]), N_fixed);
  vector[N_fixed] i_H = tail(to_vector([0.0015, 0.0345, 0.0119, 0.2329, 0.056, 1]), N_fixed);
}

parameters {
  ordered[N_peaks] E0;
  vector<lower=0>[N_peaks] gamma;
  vector<lower=0>[N_peaks] i;
  real<lower=0> Q;
  real<lower=0, upper=1> bkg;
  real<lower=0.1, upper=15> sigma;
}


model {
  E0 ~ normal(p_E0, 5*p_sigma);
  gamma ~ lognormal(0.7, 1.5);
  i ~ normal(p_i, p_i/5);



  Q ~ normal(p_Q, 10);
  bkg ~ beta(2.5, 40);

  sigma ~ normal(p_sigma, 2);
  vector[6] amplitudes = append_row(i_H, i);
  amplitudes = amplitudes/sum(amplitudes);
  
  if (prior == 0) {
    counts ~ poisson(spectrum(x_extended, x_window, sigma, bkg, m_nu, Q,
                     append_row(E_H, E0), append_row(gamma_H, gamma), amplitudes) * N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep; 
  counts_rep = poisson_rng(spectrum(x_extended, x_window,  sigma , bkg, m_nu, Q, 
                           append_row(E_H, E0), append_row(gamma_H, gamma), append_row(i_H, i)/sum(append_row(i_H, i))) * N_ev);

}

