data {
  int<lower=0> N;
  array[N] int y;
  real alpha;
  real beta;
  int<lower=0, upper=1> prior;
}

parameters {
  real<lower=0> lambda;
}

model {
  lambda ~ gamma(alpha, beta);
  if (prior == 0){
    y ~ poisson(lambda);
  }
}

generated quantities {
  array[N] int<lower = 0> y_rep = poisson_rng(rep_array(lambda, N));
  real<lower = 0> mean_y_rep = mean(to_vector(y_rep));
  real<lower = 0> sd_y_rep = sd(to_vector(y_rep));
}