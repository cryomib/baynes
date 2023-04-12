data {
  int<lower=0> N;
  array[N] int y;
  real mu;
  real sigma;
  int<lower=0, upper=1> prior;
}

transformed data {
  real<lower = 0> mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
}

parameters {
  real<lower=0> lambda;
}

model {
  lambda ~ lognormal(mu, sigma);
  y ~ poisson(lambda);
}

generated quantities {
  array[N] int<lower = 0> y_rep;
  if (prior == 0)
    y_rep = poisson_rng(rep_array(lambda, N));
  else
    y_rep = poisson_rng(rep_array(lognormal_rng(mu, sigma), N));
  real<lower = 0> mean_y_rep = mean(to_vector(y_rep));
  real<lower = 0> sd_y_rep = sd(to_vector(y_rep));
}
