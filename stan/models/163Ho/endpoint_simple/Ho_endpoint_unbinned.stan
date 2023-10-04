functions{
  #include Ho_fit_functions.stan
}

data {
  int<lower=1> N_ev;
  vector[N_ev] events;
  real<lower=0> p_Q;
  real<lower=0> p_FWHM;
  real dx;
  int<lower=0, upper=1> prior;
}

transformed data {
  real max_E = 3500;
  int N_tot = to_int(floor(max_E/dx))+1;
  vector[N_tot] E_full = linspaced_vector(N_tot, 0, N_tot-1)*dx;
  int N_peaks = 6;
  vector[N_peaks-1] E_H = tail(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks-1);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
}

parameters {
  real<lower=0> m_nu;
  real<lower=0, upper=max_E-1> Q;
  real<lower=0.1, upper=15> FWHM;
  real EM;
 //real<lower=0> sigma_ev;
 //real mu_ev;
 //vector[N_ev] z;
}

transformed parameters {
  //vector[N_ev] true_evs = mu_ev + z*sigma_ev ;
}

model {
 // m_nu ~ beta(1, 1.05);
  m_nu ~ gamma(1.135, 2.032);
  EM ~ normal(2000, 50);
  Q ~ normal(p_Q, 10);
  FWHM ~ normal(p_FWHM, 0.3);

 //mu_ev ~ normal(2700, 10);
 //sigma_ev ~ normal(200, 10);
 //z ~ std_normal();
  if (prior == 0) {
    real Ho_norm = sum(Ho_first_order(E_full, m_nu, Q, append_row(EM, E_H), gamma_H, i_H))*dx;
   // events ~ normal(true_evs, FWHM / (2 * sqrt(2 * log(2))));
    target+= sum(log(Ho_first_order(events, m_nu, Q, append_row(EM, E_H), gamma_H, i_H)))-N_ev*log(Ho_norm);
  }
}
