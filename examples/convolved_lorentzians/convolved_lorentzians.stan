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

  // gaussian window
  vector response_function(vector x, real mu, real sigma, int n) {
    vector[n] y;
    for (i in 1 : n) {
      y[i] = (1. / (sigma * (2 * pi()) ^ 0.5))
             * exp(-0.5 * (((x[i] - mu) / sigma) ^ 2));
    }
    return y;
  }

  // lorentzian
  vector lorentz(vector x, real E0, real FWHM, int n) {
    real gamma_2 = FWHM / 2.0;
    vector[n] y;
    for (i in 1 : n) {
      y[i] = gamma_2 / ((x[i] - E0) ^ 2 + gamma_2 ^ 2);
    }
    return y / pi();
  }

  vector spectrum(vector x, vector wx, vector E0, vector g, vector h,
                  real sigma, int N_peaks) {
    int Nx = num_elements(x);
    int Nwx = num_elements(wx);

    vector[Nx] y_true = rep_vector(0, Nx);
    for (k in 1 : N_peaks) {
      y_true = y_true + (h[k] .* lorentz(x, E0[k], g[k], Nx));
    }
    vector[Nwx] y_spread = response_function(wx, 0, sigma, Nwx);
    vector[Nx - Nwx + 1] y_obs = fft_convolve(y_true, y_spread);
    return y_obs / sum(y_obs);
  }
}

data {
  int<lower=1> N;
  int<lower=1> N_window;
  int<lower=1> N_peaks;
  array[N] int counts;
  vector[N] x;

  // priors parameters
  real<lower=0> p_sigma;
  vector<lower=0>[N_peaks] p_g;
  ordered[N_peaks] p_x0;
  simplex[N_peaks] p_h;
  int<lower=0, upper=1> prior;
}

transformed data {
  int N_counts = sum(counts);
  int W = N + N_window - 1;
  real dx = abs(x[2] - x[1]);

  // x axis values for a gaussian window centered in 0
  vector[N_window] window_x = linspaced_vector(N_window, 0,
                                               dx * (N_window - 1));
  window_x = window_x - dx * N_window %/% 2;

  // extend the x axis to avoid border effects in convolution
  vector[W] full_x = linspaced_vector(W, 0, dx * (W - 1));
  full_x = full_x + min(x) - dx * (N_window %/% 2);
}

parameters {
  ordered[N_peaks] x0; // center of the lorentzian
  vector<lower=0>[N_peaks] g; // FWHM of the lorentzian
  simplex[N_peaks] h; // relative heights of the peaks, their sum must be 1
  real<lower=0> sigma; // gaussian spread std
}

model {
  x0 ~ normal(p_x0, 5);
  h ~ dirichlet(p_h * 10);
  g ~ gamma(p_g, 1);
  sigma ~ gamma(p_sigma, 1);
  
  if (prior == 0) {
    counts ~ poisson(spectrum(full_x, window_x, x0, g, h, sigma, N_peaks)
                     * N_counts);
  }
}

generated quantities {
  array[N] int counts_rep;
  counts_rep = poisson_rng(spectrum(full_x, window_x, x0, g, h, sigma,
                                    N_peaks)
                           * N_counts);
}