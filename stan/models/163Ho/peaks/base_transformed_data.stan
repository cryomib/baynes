  real dx = abs(x[2] - x[1]);
  real p_sigma = 20 / (2 * sqrt(2 * log(2)));
  int N_window = to_int(floor(p_sigma * 5 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
  vector[N_bins] center_x = head(x, N_bins) + dx/2;

  int N_peaks = 5;
  vector[N_peaks] E_H = tail(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = tail(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = tail(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  real p_E0 = 2047;
  real xmin = extended_x[1];
  real xmax = extended_x[N_ext];