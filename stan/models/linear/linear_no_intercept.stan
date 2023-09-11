data {
  int<lower=0> N;       // number of cases
  vector[N] x;          // predictor (covariate)
  vector[N] y;          // outcome (variate)
  int<lower=0, upper=1> prior;
}
parameters {
  real beta;            // slope
  real<lower=0> sigma;  // outcome noise
}
model {
  beta ~ normal(0, 10);
  sigma ~ exponential(0.1);
  if (prior == 0) {
  y ~ normal(beta * x, sigma);
  }
}

generated quantities {
  array[N] real y_rep = normal_rng(beta * x, sigma);
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | x[n] * beta, sigma);
  }
}