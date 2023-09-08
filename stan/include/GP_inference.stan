  vector gp_pred_quad_rng(array[] real x2, vector y1, array[] real x1, real alpha,
                     real rho, real sigma, real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N2, N2] diag_delta;
      matrix[N1, N1] K;
      K = gp_exp_quad_cov(x1, alpha, rho);
      for (n in 1 : N1) {
        K[n, n] = K[n, n] + square(sigma);
      }
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = gp_exp_quad_cov(x1, x2, alpha, rho);
      f2_mu = k_x1_x2' * K_div_y1;
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = gp_exp_quad_cov(x2, alpha, rho) - v_pred' * v_pred;
      diag_delta = diag_matrix(rep_vector(delta, N2));
      
      f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
  }

  vector gp_pred_periodic_rng(array[] real x2, vector y1, array[] real x1,
                              real alpha, real rho, real p, real sigma,
                              real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N2, N2] diag_delta;
      matrix[N1, N1] K;
      K = gp_periodic_cov(x1, alpha, rho, p);
      for (n in 1 : N1) {
        K[n, n] = K[n, n] + square(sigma);
      }
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = gp_periodic_cov(x1, x2, alpha, rho, p);
      f2_mu = k_x1_x2' * K_div_y1;
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = gp_periodic_cov(x2, alpha, rho, p) - v_pred' * v_pred;
      diag_delta = diag_matrix(rep_vector(delta, N2));
      
      f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
  }