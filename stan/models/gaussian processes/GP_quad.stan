functions {
  #include GP_inference.stan
}
data {
  int<lower=1> N1;
  array[N1] real x1;
  vector[N1] y1;
  int<lower=1> N2;
  array[N2] real x2;
}
transformed data {
  vector[N1] mu = rep_vector(0, N1);
  real delta = 1e-9;
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}
model {
  matrix[N1, N1] L_K;
  {
    matrix[N1, N1] K = gp_exp_quad_cov(x1, alpha, rho);
    real sq_sigma = square(sigma);
    
    // diagonal elements
    for (n1 in 1 : N1) {
      K[n1, n1] = K[n1, n1] + sq_sigma;
    }
    
    L_K = cholesky_decompose(K);
  }
  
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ std_normal();
  
  y1 ~ multi_normal_cholesky(mu, L_K);
}
generated quantities {
  vector[N2] f2;
  vector[N2] y2;
  
  f2 = gp_pred_quad_rng(x2, y1, x1, alpha, rho, sigma, delta);
  for (n2 in 1 : N2) {
    y2[n2] = normal_rng(f2[n2], sigma);
  }
}

