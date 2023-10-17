  E_z ~ std_normal();
  //z ~ std_normal();
  gamma_M1 ~ gamma(13, 1);
  FWHM ~ frechet(2*p_FWHM, 0.5+p_FWHM);
  y0 ~ normal(0, 1);
  y1 ~ normal(0,1);
  A ~normal(1,0.05);
  vector[N_ext] linear =  y0 + m * (extended_x - xmin);