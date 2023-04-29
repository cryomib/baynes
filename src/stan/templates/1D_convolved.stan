functions {
  vector fft_convolve_1D(vector x, vector y) {
    int N = num_elements(x);
    int M = num_elements(y);
    int L = N + M - 1;
    int SHIFT = min(N, M);

    vector[L] x_pad = append_row(x, rep_vector(0, L - N));
    vector[L] y_pad = append_row(y, rep_vector(0, L - M));
    vector[L] full_conv = get_real(inv_fft(fft(x_pad) .* fft(y_pad)));
    return full_conv[SHIFT : L - SHIFT + 1];
  }

  vector normal_pdf(vector x, int N_x, real mu, real sigma) {
    vector[N_x] y;
    for (i in 1 : N_x) {
      y[i] = (1. / (sigma * (2 * pi()) ^ 0.5))
             * exp(-0.5 * (((x[i] - mu) / sigma) ^ 2));
    }
    return y;
  }

  vector spectrum_1D(vector x, vector wx, real sigma // additional args...
                    ) {
    int N_x = num_elements(x);
    int N_wx = num_elements(wx);

    vector[N_x] y_true = // function to be convolved with response
    vector[N_wx] y_response = normal_pdf(wx, N_wx, 0, sigma);
    vector[N_x - N_wx + 1] y_obs = fft_convolve_1D(y_true, y_response);
    return y_obs / sum(y_obs);
  }
}

data {
  int<lower=1> N;
  int<lower=1> N_window;
  array[N] int counts;
  vector[N] x;
}

transformed data {
  int N_counts = sum(counts);
  real dx = abs(x[2] - x[1]);

  // x axis values for a response window centered in 0
  vector[N_window] window_x = linspaced_vector(N_window, 0,
                                               dx * (N_window - 1));
  window_x = window_x - dx * N_window %/% 2;

  // extend the x axis to avoid border effects in convolution
  int W = N + N_window - 1;
  vector[W] full_x = linspaced_vector(W, 0, dx * (W - 1));
  full_x = full_x + min(x) - dx * (N_window %/% 2);
}