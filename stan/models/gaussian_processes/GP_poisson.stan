data {
  int<lower=1> N;
  array[N] real x;
  array[N] int<lower=0> y;
  int<lower=0, upper=1> prior;
}
transformed data {
  real delta = 1e-9;
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real a;
  vector[N] eta;
}
transformed parameters {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho);
    for (n in 1 : N) {
      K[n, n] = K[n, n] + delta;
    }
    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  a ~ std_normal();
  eta ~ std_normal();
  if (prior == 0) {
    y ~ poisson_log(f + a);
  }
}
generated quantities {
  array[N] int<lower=0> y_rep = poisson_log_rng(f + a);
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = poisson_log_lpmf(y[n] | f[n] + a);
  }
}