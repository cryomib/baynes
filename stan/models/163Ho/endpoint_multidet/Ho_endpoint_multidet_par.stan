functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  real partial_sum(array[] int y_slice,
                   int start, int end,
                   vector x_ext,
                   vector E_systs,
                   vector x_window,
                   vector FWHMs,
                   vector bkgs,
                   real m,
                   real Q,
                   vector E_H,
                   vector gamma_H,
                   vector i_H,
                   vector As,
                   vector Ns) {
    real sum = 0;
    for (i in start:end){
     sum += poisson_lpmf(y_slice[i-start+1] | spectrum(x_ext - E_systs[i], x_window, FWHMs[i], bkgs[i], m, Q, E_H, gamma_H, i_H) *As[i]* Ns[i]);
    }
    return sum;
  }

}

data {
  int<lower=1> N_bins;
  int<lower=1> N_det;

  vector[N_bins + 1] x;
  int counts[N_det, N_bins];
  real m_max;
  int N_ev[N_det];
  real<lower=0> p_Q;
  real<lower=0> FWHM[N_det];
  int<lower=0, upper=1> prior;
}

transformed data {
  real dx = abs(x[2] - x[1]);
  real p_sigma = mean(FWHM) / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 3 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;

  vector[N_window] x_window = dx * get_centered_window(N_window);
  vector[N_ext] x_extended = extend_vector(x, dx, N_window);
  vector[N_ext] bare_spectrum = Ho_lorentzians(x_extended);
  int N_peaks = 6;
  vector[N_peaks] E_H = tail(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
}

parameters {
  real<lower=0, upper=1> m_red;
  real z;
  vector[N_det] xz;
  real<lower=0> E_sigma;
  vector[N_det] E_syst;
  real<lower=0, upper=1> f_bkg[N_det];
}

transformed parameters {
  real <lower=0> Q = z * 33.54 + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  vector[N_det] A = 0.05 * xz + 1;
}

model {
  z~std_normal();
  m_red ~ beta(1, 1);
  E_sigma ~ gamma(12, 2);
  E_syst ~ normal(0, E_sigma);
  f_bkg ~ beta(1.8, 30);
  xz ~ std_normal();
    if (prior == 0) {
      target += reduce_sum(partial_sum, counts, 1,  x_extended, E_syst, x_window, FWHM, f_bkg, m_nu, Q, E_H, gamma_H, i_H, A, N_ev);
    }
}

generated quantities {
  array[N_bins] int counts1_rep = poisson_rng(spectrum(x_extended - E_syst[1], x_window, FWHM[1], f_bkg[1], m_nu, Q, E_H, gamma_H, i_H) *A[1]* N_ev[1]);
}