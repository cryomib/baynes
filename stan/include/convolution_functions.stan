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