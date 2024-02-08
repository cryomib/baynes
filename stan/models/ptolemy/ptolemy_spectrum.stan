functions {
  #include spectra.stan
}
data {
  int<lower=1> N_bins;
  vector[N_bins] x;
  array[64, 5] real coeffs;
  real Q;
  real m_nu;
}
generated quantities {
  vector[N_bins] spectrum = ptolemy(x, coeffs, m_nu, Q);
  vector[N_bins] erfu = erf(x);
}
