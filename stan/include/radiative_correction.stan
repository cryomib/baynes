  real spence(real x) {
    real PI = 3.1415926535897932;
    array[6] real P = {0.9999999999999999502, -2.6883926818565423430,
                       2.6477222699473109692, -1.1538559607887416355,
                       2.0886077795020607837e-1, -1.0859777134152463084e-2};

    array[7] real Q = {1.0000000000000000000, -2.9383926818565635485,
                       3.2712093293018635389, -1.7076702173954289421,
                       4.1596017228400603836e-1, -3.9801343754084482956e-2,
                       8.2743668974466659035e-4};
    real y = 0, r = 0, s = 1;
    if (x < -1) {
      real l = log(1 - x);
      y = 1 / (1 - x);
      r = -PI * PI / 6 + l * (0.5 * l - log(-x));
      s = 1;
    } else if (x == -1) {
      return -PI * PI / 12;
    } else if (x < 0) {
      real l = log1p(-x);
      y = x / (x - 1);
      r = -0.5 * l * l;
      s = -1;
    } else if (x == 0) {
      return 0;
    } else if (x < 0.5) {
      y = x;
      r = 0;
      s = 1;
    } else if (x < 1) {
      y = 1 - x;
      r = PI * PI / 6 - log(x) * log(y);
      s = -1;
    } else if (x == 1) {
      return PI * PI / 6;
    } else if (x < 2) {
      real l = log(x);
      y = 1 - 1 / x;
      r = PI * PI / 6 - l * (log(y) + 0.5 * l);
      s = 1;
    } else {
      real l = log(x);
      y = 1 / x;
      r = PI * PI / 3 - 0.5 * l * l;
      s = -1;
    }
    real y2 = y * y;
    real y4 = y2 * y2;
    real p = P[0] + y * P[1] + y2 * (P[2] + y * P[3])
             + y4 * (P[4] + y * P[5]);
    real q = Q[0] + y * Q[1] + y2 * (Q[2] + y * Q[3])
             + y4 * (Q[4] + y * Q[5] + y2 * Q[6]);
    return -(r + s * y * p / q);
  }

  vector spence(vector x){
    int N = num_elements(x);
    vector[N] y = rep_vector(0, N);
    for (i in 1:N){
        y[i] = spence(x[i]);
    }
    return y;
  }
