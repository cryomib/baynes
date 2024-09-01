  vector interp1d(vector x, vector y, vector x_new) {
    // Perform linear interpolation between data points of monotonically increasing vectors.
    int N = num_elements(x);
    int N_new = num_elements(x_new);
    vector[N_new] y_new;
    int j = 1;
    for (i in 1 : N) {
      if (x_new[i] <= x[1]) {
        y_new[i] = y[1];
      } else if (x_new[i] >= x[N]) {
        y_new[i] = y[N];
      } else {
        while (j < N - 1 && x_new[i] > x[j + 1]) {
            j += 1;
          }
        y_new[i] = y[j]
                   + (y[j + 1] - y[j]) / (x[j + 1] - x[j])
                     * (x_new[i] - x[j]);
      }
    }
    return y_new;
  }
