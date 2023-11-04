data {
  int<lower=0> N;
  vector[N] x_meas;
  vector[N] y;
  real<lower=0> tau;
  real<lower=0> sigma;
  int<lower=0, upper=1> prior;
}
parameters {
  real alpha;
  real beta;
  vector[N] x_true;
  real mu_x;
  real sigma_x;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  sigma ~ exponential(5);
  mu_x ~ normal(10, 10);
  sigma_x ~ normal(1,2);

  x_true ~ normal(mu_x, sigma_x);

  if (prior == 0) {
    x_meas ~ normal(x_true, tau);
    y ~ normal(alpha + beta * x_true, sigma);
  }
}

generated quantities {
   array[N] real y_rep = normal_rng(alpha + beta * x_true, sigma);
}