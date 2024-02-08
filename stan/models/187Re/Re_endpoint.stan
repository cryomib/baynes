functions{
  #include convolution_functions.stan
  #include Ho_fit_functions.stan
  vector re_bare_spectrum(vector E){
    real me = 510998;
    return sqrt((E+me)^2 - me^2) * E * (19.95./E + 1 − 6.80e-6 * E +3.05e-9  * E^2 );
  }
}

data {
  int<lower=1> N_bins;
  vector[N_bins + 1] x;
  array[N_bins] int counts;
  real m_max;
  real<lower=0> N_ev;
  real<lower=0> p_Q;
  real<lower=0> p_std_Q;
  real<lower=0> p_FWHM;
  real<lower=0> p_FWHM_std;
  int<lower=0, upper=1> prior;
}

transformed data {
 real p_sigma = (p_FWHM+p_FWHM_std) / (2 * sqrt(2 * log(2)));
 real dx = abs(x[2] - x[1]);
 int N_window = to_int(floor(p_sigma * 3 / dx)) * 2 + 1;
 int N_ext = num_elements(x) + N_window - 1;
 vector[N_bins] centers_x = head(x, N_bins) + dx/2;
 vector[N_window] window_x = dx * get_centered_window(N_window);
 vector[N_ext] extended_x = extend_vector(x, dx, N_window);
 vector[N_ext] bare_spectrum = re_bare_spectrum(extended_x);
}

parameters {
  real<lower=0, upper=1> m_red;
  real z;
  real xz;
  real<lower=0, upper=1> f_bkg;
  real<lower=0.1, upper=p_FWHM*3+p_FWHM_std> FWHM;
}

transformed parameters {
  real <lower=0> Q = z + p_Q;
  real<lower=0, upper=m_max> m_nu = m_red * m_max;
  real <lower=0> A = 0.1 * xz + 1;
}

model {
  m_red ~ beta(1, 1);
  z ~ normal(0, p_std_Q);
  xz~std_normal();
  f_bkg ~ beta(1.8, 30);
  FWHM ~ normal(p_FWHM, p_FWHM_std);
  if (prior == 0) {
    counts ~ poisson(spectrum(extended_x, window_x, FWHM, f_bkg, m_nu, Q, bare_spectrum) *A* N_ev);
  }
}

generated quantities {
  array[N_bins] int counts_rep = poisson_rng(spectrum(extended_x, window_x, FWHM, f_bkg, m_nu, Q, bare_spectrum) *A* N_ev);
}
