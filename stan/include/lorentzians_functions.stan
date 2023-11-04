  vector lorentzian(vector E, real E0, real gamma){
    return gamma/2*pi() * 1./((E-E0)^2 + (gamma/2)^2);
  }

  vector lorentzian_asymm(vector E, real E0, real FWHM, real asymm) {
    real gamma_2 = FWHM / 2.0;
    real gamma_L = (1-asymm) * gamma_2;
    real gamma_R = (1+asymm) * gamma_2;

    int N = num_elements(E);
    vector[N] y;
    for (i in 1 : N) {
      if (E[i] < E0){
        y[i] = gamma_L / ((E[i] - E0) ^ 2 + gamma_L ^ 2);
      }
      else{
        y[i] = gamma_L / ((E[i] - E0) ^ 2 + gamma_L ^ 2);
      }
    }
    real norm1 = 1/(2 * gamma_L);
    real norm2 = 1/(2 * gamma_R);
    return y *(norm1+norm2)/ pi();
  }

  vector sigmoid(vector x, real a, real b, real gamma){
    return 2*gamma/(1+exp(-a*(x-b)));
  }


  vector lorentzian_xps(vector E, real E0, real FWHM, real a, real b){
    int N = num_elements(E);
    vector[N] x = E-E0;
    vector[N] sigmoid_2 = sigmoid(x, a, b, FWHM)/2;
    return 1/pi() * sigmoid_2./(sigmoid_2^2+x^2);
  }

vector lorentzians(vector E, vector E0s, vector gammas, vector is){
    int N = num_elements(E);
    int N_peaks = num_elements(E0s);
      vector[N] lors = rep_vector(0, N);
    for (i in 1 : N){
      lors[i] = dot_product(is, gammas./((E[i] - E0s) ^ 2 + (gammas/2) ^ 2));
    }
    return lors;
  }

  vector spectrum(vector x_full, vector x_window, real sigma, real bkg, vector E0s, vector gammas, vector is){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = lorentzians(x_full, E0s, gammas, is);
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = rep_vector(bkg, (Nx-1)) + y_centers;
    if (sigma == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, sigma, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }

  vector spectrum(vector x_full, vector x_window, real sigma, vector E0s, vector gammas, vector asymms, vector is){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    int N_peaks = num_elements(E0s);
    vector[Nx] y_true = rep_vector(0, Nx);
    for (i in 1: N_peaks){
      y_true = y_true + is[i]*lorentzian_asymm(x_full, E0s[i], gammas[i], asymms[i]);
    }
    vector[Nx-1] y_full = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    if (sigma == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, sigma, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }