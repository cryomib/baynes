functions {

  // gaussian
  vector normal_pdf(vector x, int N_x, real mu, real sigma) {
    vector[N_x] y;
    for (i in 1 : N_x) {
      y[i] = (1. / (sigma * (2 * pi()) ^ 0.5))
             * exp(-0.5 * (((x[i] - mu) / sigma) ^ 2));
    }
    return y;
  }

  // lorentzian
  vector lorentz_pdf(vector x, int N_x, real E0, real FWHM) {
    real gamma_2 = FWHM / 2.0;
    vector[N_x] y;
    for (i in 1 : N_x) {
      y[i] = gamma_2 / ((x[i] - E0) ^ 2 + gamma_2 ^ 2);
    }
    return y / pi();
  }
  
}