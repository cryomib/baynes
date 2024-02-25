functions{
  #include convolution_functions.stan
}

data {
  int<lower=1> Nx;
  vector[Nx] x;

  int<lower=1> Ny;
  vector[Ny] y;
}


generated quantities {
  vector[Nx-Ny+1] conv = fft_convolve(x,y); 
}

