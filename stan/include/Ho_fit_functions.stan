  // First order 163Ho EC decay spectrum
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

  // 163Ho lorentzian peaks with default values
  vector Ho_lorentzians(vector E){
    int N = num_elements(E);
    int N_peaks = 6;
    vector[N_peaks] E_H = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
    vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
    vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
    i_H = i_H/sum(i_H);
    vector[N] lorentzians = rep_vector(0, N);
    for (i in 1 : N){
      lorentzians[i] = dot_product(i_H, gamma_H./((E[i] - E_H) ^ 2 + (gamma_H/2) ^ 2));
    }
    return lorentzians;
  }

  vector Ho_lorentzians(vector E, vector E_H, vector gamma_H, vector i_H){
    int N = num_elements(E);
    vector[N] lorentzians = rep_vector(0, N);
    for (i in 1 : N){
      lorentzians[i] = dot_product(i_H, gamma_H./((E[i] - E_H) ^ 2 + (gamma_H/2) ^ 2));
    }
    return lorentzians;
  }



  // 163Ho pileup spectrum
  vector Ho_pileup(data vector x, data real dx, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H){
    int N = num_elements(x);
    int N_ext = to_int(floor(x[N]/dx))+1;
    vector[N_ext] x_pu = linspaced_vector(N_ext, 0, (N_ext-1))*dx;
    vector[N_ext] Ho_edges = Ho_first_order(x_pu, m_nu, Q_H, E_H, gamma_H, i_H);
    vector[N_ext-1] Ho_centers = (head(Ho_edges, N_ext-1) + tail(Ho_edges, N_ext-1))/2;
    return segment(autocorrelation(Ho_centers), N_ext-N+1, N-1);
  }

  // convolve true spectrum true_y with response function
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

  // Fixed bare spectrum (without phase space), no background
  vector spectrum(vector x_full, vector x_window, real FWHM, real m_nu, real Q_H, vector bare_spectrum){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = rep_vector(0, Nx);
    for (i in 1 : Nx){
      if (Q_H - m_nu - x_full[i] < 0){
        y_true[i] = 0;
      }
      else{
        y_true[i] = bare_spectrum[i] * (Q_H - x_full[i]) * sqrt((Q_H - x_full[i])^2-m_nu^2)/(2*pi());
      }
    }
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    return convolve_spectrum(x_window, y_centers, FWHM);
  }

  // Fixed bare spectrum (without phase space), flat background
  vector spectrum(vector x_full, vector x_window, real FWHM, real p_bkg, real m_nu, real Q_H, vector bare_spectrum){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = rep_vector(0, Nx);
    for (i in 1 : Nx){
      if (Q_H - m_nu - x_full[i] < 0){
        y_true[i] = 0;
      }
      else{
        y_true[i] = bare_spectrum[i] * (Q_H - x_full[i]) * sqrt((Q_H - x_full[i])^2-m_nu^2)/(2*pi());
      }
    }
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = (p_bkg/(Nx-Nwx)) +
                          ((1-p_bkg) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));

    return convolve_spectrum(x_window, y_full, FWHM);
  }

  // compute full spectrum, flat background
  vector spectrum(vector x_full, vector x_window, real FWHM, real p_bkg, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = Ho_first_order(x_full, m_nu, Q_H, E_H, gamma_H, i_H);
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = (p_bkg/(Nx-Nwx))+
                          ((1-p_bkg) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));
    if (FWHM == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, FWHM, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }

  // fixed bare spectrum, flat background and fixed pileup
  vector spectrum(vector x_full, vector x_window, real FWHM, real p_bkg, real p_pu, vector pu_spectrum, real m_nu, real Q_H, vector bare_spectrum){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = rep_vector(0, Nx);
    for (i in 1 : Nx){
      if (Q_H - m_nu - x_full[i] < 0){
        y_true[i] = 0;
      }
      else{
        y_true[i] = bare_spectrum[i] * (Q_H - x_full[i]) * sqrt((Q_H - x_full[i])^2-m_nu^2)/(2*pi());
      }
    }
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = (p_bkg/(Nx-Nwx)) +
                          (p_pu * pu_spectrum / sum(segment(pu_spectrum, Nwx%/%2 +1, Nx-Nwx))) +
                          ((1-p_bkg-p_pu) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));
    if (FWHM == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, FWHM, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }

  // compute full spectrum, flat background and fixed pileup spectrum
  vector spectrum(vector x_full, vector x_window, real FWHM, real p_bkg, real p_pu, vector pu_spectrum, real m_nu, real Q_H, vector E_H, vector gamma_H, vector i_H){
    int Nx = num_elements(x_full);
    int Nwx = num_elements(x_window);
    vector[Nx] y_true = Ho_first_order(x_full, m_nu, Q_H, E_H, gamma_H, i_H);
    vector[Nx-1] y_centers = (head(y_true, Nx-1) + tail(y_true, Nx-1))/2;
    vector[Nx-1] y_full = (p_bkg/(Nx-Nwx)) +
                          (p_pu * pu_spectrum / sum(segment(pu_spectrum, Nwx%/%2 +1, Nx-Nwx))) +
                          ((1-p_bkg-p_pu) * y_centers / sum(segment(y_centers, Nwx%/%2 +1, Nx-Nwx)));
    if (FWHM == 0){
      return y_full;
    }
    else{
      vector[Nwx] y_spread = gaussian_response(x_window, FWHM, Nwx);
      vector[Nx - Nwx] y_obs = fft_convolve(y_full, y_spread);
      return y_obs / sum(y_obs);
    }
  }