data {
  int<lower=0> N;
  real alpha;
  real beta;
  real alpha_true;
  real beta_true;
}
transformed data {
  real<lower = 0> lambda_sim = gamma_rng(alpha_true, beta_true);
  array[N] int<lower = 0> y = poisson_rng(rep_array(lambda_sim, N));
}
parameters {
  real<lower=0> lambda;
}
model {
  lambda ~ gamma(alpha, beta);
  y ~ poisson(lambda);
}
generated quantities {
  int<lower = 0, upper = 1> lt_lambda  = lambda < lambda_sim;
}