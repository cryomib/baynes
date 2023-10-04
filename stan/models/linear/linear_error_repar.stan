data {
  int<lower=0> N;       // number of cases
  vector[N] x_meas;          // predictor (covariate)
  vector[N] y; 
  real<lower=0> tau;     // measurement noise
  real<lower=0> sigma;  // outcome noise
  int<lower=0, upper=1> prior;
}
parameters {
  real alpha;           // intercept
  real beta;            // slope
  vector[N] z;    // unknown true value
  real mu_x;          // prior location
  real sigma_x;       // prior scale
}
transformed parameters {
   vector[N] x_true = z * sigma_x + mu_x;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  sigma ~ gamma(4,1);
  mu_x ~ normal(10, 10);
  sigma_x ~ normal(1,2);

  z ~ normal(rep_vector(0, N), 1);
  //x_true ~ normal(mu_x, sigma_x);  // prior

  if (prior == 0) {
    x_meas ~ normal(x_true, tau);    // measurement model
    y ~ normal(alpha + beta * x_true, sigma);
  }
}

//generated quantities {
//   array[N] real y_rep = normal_rng(alpha + beta * x_true, sigma);
//}