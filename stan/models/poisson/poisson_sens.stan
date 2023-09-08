data {
  int<lower=0> N;
  array[N] int y;
  int<lower=0, upper=1> prior;
}

parameters {
  real<lower=0> lambda;
  real<lower=0> mu;
}

model {
  mu ~ normal(10, 20);
  lambda ~ gamma(mu/5, mu^2/5);
  if (prior == 0){
    y ~ poisson(lambda);
  }
}

generated quantities {
  array[N] int<lower = 0> y_rep = poisson_rng(rep_array(lambda, N));
  real<lower = 0> mean_y_rep = mean(to_vector(y_rep));
  real<lower = 0> sd_y_rep = sd(to_vector(y_rep));
}