parameters {
  real y;
  real z;
}
transformed parameters {
  real x;
  x = z * exp(y);
}
model {
  y  ~ normal(0, 3);
  z ~ std_normal();
}
