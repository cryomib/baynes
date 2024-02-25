data {
  int<lower=0> N;
  vector[N] x;          // predictor
  vector[N] y;          // outcome
  vector[N] sigma;          // outcome
  int<lower=0, upper=1> prior;
}
parameters {
  real alpha;           // intercept
  real beta;            // slope
  real<lower=0> sigma_add;  // outcome noise
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ normal(0, 2);
  if (prior == 0) {
  y ~ normal(alpha + beta * x, sigma+sigma_add);
  }
}

generated quantities {
  array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | alpha + x[n] * beta, sigma);
  }
}