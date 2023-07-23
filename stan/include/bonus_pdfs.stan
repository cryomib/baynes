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
  
  // first order 163Ho spectrum
  vector Ho_first_order(vector E, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H){
    int N = num_elements(E);
    vector[N] Ho_spectrum = rep_vector(0, N);
    for (i in 1 : N){
      if (Q_H - m_nu - E[i] < 0){
        Ho_spectrum[i] = 0;
      }
      else{
        Ho_spectrum[i] = dot_product(i_H, gamma_H./((E[i] - E_H) ^ 2 + (gamma_H/2) ^ 2));
        Ho_spectrum[i] = Ho_spectrum[i] * (Q_H - E[i]) * sqrt((Q_H - E[i])^2-m_nu^2)/(2*pi());
      }

    }
    return Ho_spectrum;
  }
}