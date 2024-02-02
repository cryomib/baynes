functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  #include spectra.stan
}

data {
  int<lower=1> N_bins;
  vector[N_bins+1] x;
  array[N_bins] int counts;
  array[64, 5] real coeffs;
  real m_max;
  real<lower=0> p_Q;
  real<lower=0> p_std_Q;
    real<lower=0> p_std_Qs;

  real<lower=0> p_FWHM;
  real<lower=0> p_std_FWHM;
  int<lower=0, upper=1> prior;
  int N_ev;

}

transformed data {
  real p_sigma = (p_FWHM+p_std_FWHM) / (2 * sqrt(2 * log(2)));
  real dx = abs(x[2] - x[1]);
  int N_window = to_int(floor(p_sigma * 3 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_bins] centers_x = head(x, N_bins) + dx/2;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
  real bkg_norm = dx * N_ext;
}

parameters {
  real<lower=0, upper=1> m_red;
  real z;
  positive_ordered[63] Qs; 
 // real xz;
}

transformed parameters {
  real <lower=0> Q = z*p_std_Q + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  //real <lower=0> A = 0.1 * xz + 1;
}

model {

  m_red ~ beta(1, 1);
  z ~ std_normal();
 // xz ~ std_normal();
  array[64, 5] real q_coeffs = coeffs;
  for (j in 1:63){
    Qs[j]~normal(coeffs[j+1][1], p_std_Qs);
    q_coeffs[j+1][1] = Qs[j];
  }


  if (prior == 0) {
    vector[N_ext] spectrum = ptolemy(extended_x, q_coeffs, m_nu, Q);
    vector[N_bins] convolved = convolve_and_bin(window_x, spectrum, p_FWHM);
   // counts ~ poisson(((1-f_bkg)*bare_spectrum + f_bkg * bkg_norm) *A* N_ev);
    counts ~ poisson(convolved*N_ev + 1e-6);

  }
}

generated quantities {
  array[N_bins] int counts_rep;
  {
    vector[N_ext] spec = ptolemy(extended_x, coeffs, m_nu, Q);
    vector[N_bins] convolved = convolve_and_bin(window_x, spec, p_FWHM);
    counts_rep = poisson_rng(convolved * N_ev+1e-4);

  }
}
