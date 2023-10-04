data {
  int<lower=0> N;       // number of cases
  vector[N] x_meas;          // predictor (covariate)
  int<lower=0, upper=1> prior;
}
parameters {
  real<lower=0> tau;         // slope
  vector<lower=0>[N] x_true;    // unknown true value
  real<lower=0> mu_x;          // prior location
  real<lower=0> sigma_x;       // prior scale
}
model {
  tau ~ gamma(2, 1);
  mu_x ~ gamma(5, 1);
  sigma_x ~ gamma(1,1);

  x_true ~ gamma(rep_vector(mu_x, N), sigma_x);  // prior

  if (prior == 0) {
    x_meas ~ normal(x_true, tau);
  }
}

generated quantities {
   vector[N] y_rep = to_vector(normal_rng(x_true, tau));
}