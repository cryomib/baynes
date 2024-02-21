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

  vector gaussian_response(vector x, real FWHM, int n) {
    vector[n] y = exp(-4 * log(2) * ((x / FWHM) ^ 2));
    return y/sum(y);
  }

  vector get_centered_window(int N_window){
    return linspaced_vector(N_window, - N_window %/% 2, N_window %/% 2);
  }

  vector extend_vector(data vector x, data real dx, data int N_window){
    int N_ext = num_elements(x) + N_window - 1;
    vector[N_ext] x_ext = linspaced_vector(N_ext, 0., dx * (N_ext - 1));
    x_ext = x_ext + min(x) - dx * (N_window %/% 2);
    return x_ext;
  }

  // convolve true spectrum true_y with response function considering bin centers
  vector convolve_spectrum(vector x_window, vector y_full, real FWHM){
    int Ny = num_elements(y_full);
    int Nwx = num_elements(x_window);
    if (FWHM == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, FWHM, Nwx);
      vector[Ny - Nwx + 1] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }

  // convolve true spectrum true_y with gaussian response function from bin edges
  vector convolve_and_bin(vector x_window, vector y_full, real FWHM){
    int Ny = num_elements(y_full);
    int Nwx = num_elements(x_window);
    if (FWHM == 0){
      return (head(y_full, Ny-1) + tail(y_full, Ny-1))/2;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, FWHM, Nwx);
      vector[Ny - Nwx + 1] y_obs = fft_convolve(y_full, y_spread);
      vector[Ny - Nwx] y_centers =  (head(y_obs, Ny-Nwx) + tail(y_obs, Ny-Nwx))/2;
      return y_centers / sum(y_centers);
    }
  }

  // convolve true spectrum true_y with response function from bin edges
  vector convolve_and_bin(vector y_full, vector y_response){
    int Ny = num_elements(y_full);
    int Nwx = num_elements(y_response);
    vector[Ny - Nwx + 1] y_obs = fft_convolve(y_full, y_response);
    vector[Ny - Nwx] y_centers =  (head(y_obs, Ny-Nwx) + tail(y_obs, Ny-Nwx))/2;
    return y_centers / sum(y_centers);
  }
