  real dx = abs(x[2] - x[1]);
  int N_window = to_int(floor(p_sigma * 3 / dx)) * 2 + 1;
  int N_ext = num_elements(x) + N_window - 1;
  vector[N_bins] centers_x = head(x, N_bins) + dx/2;

  vector[N_window] window_x = dx * get_centered_window(N_window);
  vector[N_ext] extended_x = extend_vector(x, dx, N_window);
  //vector[N_ext] bare_spectrum = Ho_lorentzians(extended_x);

  int N_peaks = 6;
  vector[N_peaks] E_H = head(to_vector([2047, 1842, 414.2, 333.5, 49.9, 26.3]), N_peaks);
  vector[N_peaks] gamma_H = head(to_vector([13.2, 6.0, 5.4, 5.3, 3.0, 3.0]), N_peaks);
  vector[N_peaks] i_H = head(to_vector([1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015]), N_peaks);
  //vector[N_ext-1] pu_spectrum = Ho_pileup(extended_x, dx, 0, p_Q, E_H, gamma_H, i_H);