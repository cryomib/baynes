functions {
  vector autocorrelation(vector x){
    int N = num_elements(x);
    int N_full = 2 * N - 1;
    vector[N_full] x_pad = append_row(x, rep_vector(0, N - 1));
    complex_vector[N_full] x_fft = fft(x_pad);
    return get_real(inv_fft(x_fft.*x_fft));
    }

  vector fft_convolve(vector x, vector y){
    int N = num_elements(x);
    int M = num_elements(y);
    int L = N + M - 1;
    int SHIFT = min(N, M);
    vector[L] x_pad = append_row(x, rep_vector(0, L - N));
    vector[L] y_pad = append_row(y, rep_vector(0, L - M));
    vector[L] full_conv = get_real(inv_fft(fft(x_pad) .* fft(y_pad)));

    return full_conv[SHIFT : L - SHIFT + 1];
  }

  vector gaussian_response(vector x, real sigma, int n) {
    vector[n] y = exp(-0.5 * ((x / sigma) ^ 2));
    return y/sum(y);
  }

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

  vector spectrum(vector full_x, vector window_x, real sigma, real p_bkg, real p_pu, vector pu_spectrum, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H, int n_peaks){
    int Nx = num_elements(full_x);
    int Nwx = num_elements(window_x);
    vector[Nx] y_true = Ho_first_order(full_x, m_nu, Q_H, E_H, gamma_H, i_H);
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = rep_vector(p_bkg/(Nx-Nwx), (Nx-1)) + 
                          (p_pu * pu_spectrum / sum(segment(pu_spectrum, Nwx%/%2 +1, Nx-Nwx))) +
                          ((1-p_bkg-p_pu) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));
    if (sigma == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(window_x, sigma, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    } 
  }

    vector spectrum(vector full_x, vector window_x, real sigma, real p_bkg, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H, int n_peaks){
    int Nx = num_elements(full_x);
    int Nwx = num_elements(window_x);
    vector[Nx] y_true = Ho_first_order(full_x, m_nu, Q_H, E_H, gamma_H, i_H);
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = rep_vector(p_bkg/(Nx-Nwx), (Nx-1)) + 
                          ((1-p_bkg) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));
    if (sigma == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(window_x, sigma, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    } 
  }
}