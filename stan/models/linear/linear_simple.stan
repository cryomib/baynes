data {
  int<lower=0> N;       // number of cases
  vector[N] x;          // predictor (covariate)
  vector[N] y;          // outcome (variate)
  int<lower=0, upper=1> prior;
}
parameters {
  real alpha;           // intercept
  real beta;            // slope
  real<lower=0> sigma;  // outcome noise
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  if (prior == 0) {
  y ~ normal(alpha + beta * x, sigma);
  }
}

generated quantities {
   array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
}