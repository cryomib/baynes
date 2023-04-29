functions {
  
  // 1D (vector) fast convolution with Fourier transform, equivalent to "same" option of scipy's convolve 
  vector fft_convolve_1D(vector x, vector y) {
    int N = num_elements(x);
    int M = num_elements(y);
    int L = N + M - 1;
    int SHIFT = min(N, M);

    vector[L] x_padded = append_row(x, rep_vector(0, L - N));
    vector[L] y_padded = append_row(y, rep_vector(0, L - M));
    vector[L] full_conv = get_real(inv_fft(fft(x_padded) .* fft(y_padded)));
    return full_conv[SHIFT : L - SHIFT + 1];
  }

  // 1D (vector) template for convolved spectra
  vector spectrum_1D(vector x, vector wx, // additional args...
                    ) {
    int N_x = num_elements(x);
    int N_wx = num_elements(wx);

    vector[N_x] y_true = 
    vector[N_wx] y_response = 
    vector[N_x - N_wx + 1] y_obs = fft_convolve_1D(y_true, y_response);
    return y_obs / sum(y_obs);
  }

}