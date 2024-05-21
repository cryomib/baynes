data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] x;          // predictor
  array[M, N] real y;          // outcome
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
  sigma ~ exponential(0.1);
  if (prior == 0) {
  for(i in 1:M){
  y[i] ~ normal(alpha + beta * (x-x[1]), sigma);}
  }
}

generated quantities {
  array[N] real y_rep = normal_rng(alpha + beta * (x-x[1]), sigma);
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | alpha + (x[n]-x[1])* beta, sigma);
  }
}