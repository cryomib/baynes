data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
  real<lower=0> nu;
  int<lower=0, upper=1> prior;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  if (prior==0){
    y ~ student_t(nu, alpha + beta * x, sigma);
  }
}
generated quantities {
  array[N] real y_rep = student_t_rng( nu, alpha + beta * x, sigma);
}