  #include convolution_functions.stan

  vector lorentzians(vector E, vector E0s, vector gammas, vector is){
    int N = num_elements(E);
    int N_peaks = 6;
      vector[N] lorentzians = rep_vector(0, N);
    for (i in 1 : N){
      lorentzians[i] = dot_product(is, gammas./((E[i] - E0s) ^ 2 + (gammas/2) ^ 2));
    }
    return lorentzians;
  }

  vector spectrum(vector x_full, vector x_window, real sigma, real p_bkg, vector E0s, vector gammas, vector is){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = lorentzians(x_full, E0s, gammas, is);
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = rep_vector(p_bkg/(Nx-Nwx), (Nx-1)) +
                          ((1-p_bkg) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));
    if (sigma == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, sigma, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }