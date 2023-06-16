functions {
    vector fft_convolve(vector x, vector y) {
    int N = num_elements(x);
    int M = num_elements(y);
    int L = N + M - 1;
    int SHIFT = min(N, M);
    vector[L] x_pad = append_row(x, rep_vector(0, L - N));
    vector[L] y_pad = append_row(y, rep_vector(0, L - M));
    vector[L] full_conv = get_real(inv_fft(fft(x_pad) .* fft(y_pad)));

    return full_conv[SHIFT : L - SHIFT + 1];
  }

  vector response_function(vector x, real sigma, int n) {
    vector[n] y = exp(-0.5 * ((x / sigma) ^ 2));
    return y/sum(y);
  }

  vector spectrum(vector full_x, vector window_x, real sigma, real p_bkg, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H, int n_peaks){
    int Nx = num_elements(full_x);
    int Nwx = num_elements(window_x);
    vector[Nx] y_true = rep_vector(0, Nx);
    for (i in 1 : Nx){
      if (Q_H - m_nu - full_x[i] < 0){
        y_true[i] = 0;
      }
      else{
        for (k in 1 : n_peaks){
          y_true[i] += i_H[k] * gamma_H[k]/((full_x[i] - E_H[k]) ^ 2 + (gamma_H[k]/2) ^ 2);
        }
        y_true[i] = y_true[i] * (Q_H - full_x[i]) * sqrt((Q_H - full_x[i])^2-m_nu^2)/(2*pi()); 
      }
    }
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    
    vector[Nwx] y_spread = response_function(window_x, sigma, Nwx);
    vector[Nx - Nwx] y_obs = fft_convolve(y_centers, y_spread);
    y_obs = rep_vector(p_bkg/(Nx-Nwx), (Nx-Nwx)) + ((1-p_bkg) * y_obs / sum(y_obs));

    return y_obs / sum(y_obs);
  }

  vector lorentz(vector x, real E0, real FWHM, int n) {
    real gamma_2 = FWHM / 2.0;
    vector[n] y;
    for (i in 1 : n) {
      y[i] = gamma_2 / ((x[i] - E0) ^ 2 + gamma_2 ^ 2);
    }
    return y / pi();
  }

  real HoSpectrumFirstOrder(real E, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H, int n_peaks){
    real y = 0;
    if (Q_H - m_nu - E > 0){
      for (k in 1 : n_peaks){
        y += i_H[k] * gamma_H[k]/((E - E_H[k]) ^ 2 + (gamma_H[k]/2) ^ 2);
      }
      y *= (Q_H - E) * sqrt((Q_H - E)^2-m_nu^2)/(2*pi()); 
    }
    return y;
  }
}