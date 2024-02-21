vector ptolemy(vector E, array[,] real coeffs, real m_nu, real Q){
    int N = num_elements(E);
    vector[N] y = rep_vector(0, N);

    real me = 510998.95;
    real mhe3 = 2809413505.67592;
    real lambda = 4.21e-5;
    real eps0 = 5.76;
    vector[N] pb = sqrt(E^2 + 2*E*me);


    real i_disc = lambda /(2 * pi()^3);
    real E_nu, p_nu, xb;
    real Qn, pn, an, bn, cn;
    for (k in 1:64)
    {
        Qn = Q - coeffs[k][1];
        pn = coeffs[k][2];
        an = coeffs[k][3];
        bn = coeffs[k][4];
        cn = coeffs[k][5];
        for (j in 1:N)
        {
            E_nu = Qn - E[j];
            if (E_nu >= m_nu){
                xb = (pb[j] - pn) / pn;
                y[j] = y[j] + i_disc * pb[j] * (E[j] + me) * sqrt(E_nu^2 - m_nu^2) * E_nu *
                              (an + xb*lambda*pn*(2*bn - lambda*pn*cn));
            }
        }
    }


    real i_cont = 1. / (pi()^(7.0/2));
    real QKE, kinf2, pb2_2mhe3;
    real b, bl, pl, exppl, expbl, erfplbl;
    real I0, I1, I2, I3, I4, I5;
    for (j in 1:N)
    {
        QKE = Q - E[j] - eps0;
        if(QKE>=m_nu) {
            b = -pb[j] + sqrt(2*mhe3*(QKE - m_nu));
            bl = b * lambda;
            pl = pb[j] * lambda;
            expbl = exp(-bl^2);
            exppl = exp(-pl^2);
            erfplbl= (erf(pl) + erf(bl));

            pb2_2mhe3 = pb[j]^2/(2*mhe3);
            kinf2 = sqrt(1 - (m_nu/(4.0 + QKE - pb2_2mhe3))^2);

            I0 = sqrt(pi())/(2*lambda) * pb[j] * (QKE - pb2_2mhe3)^2 * erfplbl;
            I1 = 1. /(2*lambda^2) * (QKE - pb2_2mhe3)*(QKE - 5.0*pb2_2mhe3) * (exppl - expbl);
            I2 = 1. / (4* lambda^3) * ((5.0 * pb2_2mhe3 - 3.0*QKE) * pb[j] / mhe3) * (-2*pl *exppl-2*bl * expbl+sqrt(pi())*erfplbl);
            I3 = 1. /(2*lambda^4) *  ((5.0*pb2_2mhe3 - QKE) / mhe3) * (exppl*(1+pl^2) - expbl *(1+bl^2));
            I4 = 5.*pb[j]/(32 * mhe3^2 * lambda^5) * (-expbl*(6*bl+4*bl^3) - exppl*(6.0*pl+4*pl^3) + 3*sqrt(pi())*erfplbl);
            I5 = 1. / (8*mhe3^2*lambda^6) * (exppl*(2+2*pl^2+pl^4) - expbl*(2.0+2.0*bl^2+bl^4));
            y[j] = y[j] + i_cont * (E[j] + me) * kinf2 * (I0 + I1 + I2 + I3 + I4 + I5);
        }
    }
    return y;
    }


vector allowed_beta(vector E, real m_nu, real Q){
    int N = num_elements(E);
    vector[N] y = rep_vector(0, N);

    real me = 510998.95;
    real pb;
    real beta_val;
    real eta;
    real F;

    for (j in 1:N){
        if (Q-E[j]>=m_nu){
            pb = sqrt(E[j]^2 + 2*E[j]*me);
            beta_val = pb/(E[j]+me);
            eta = 2.0*0.04585061813815046 / beta_val;
            F = eta * (1.002037 - 0.001427 * beta_val) / (1 - exp(-eta));
            y[j] = y[j] + pb * (E[j]+me)*(Q-E[j])*sqrt((Q - E[j]) ^ 2 - m_nu^2);

        }
    }
    return y;

}

vector Re187(vector E, real m_nu, real Q){
    int N = num_elements(E);
    vector[N] y = rep_vector(0, N);

    real bm1 = 19.5,  b1 = -6.8e-6,  b2 = 3.05e-9;
    real fd1 = 3.01258188e02, fd2 = -4.98343890e-01, fd3 = 5.69632611e-04;
    real me = 510998.95;

    real pb;
    real FD;
    real exchange;

    for (i in 1:N) {
        if (Q - E[i] >= m_nu) {
            pb = sqrt(E[i]^2 + 2 * E[i] * me);
            FD = exp(
                log(fd1) +
                fd2 * log(E[i]) +
                fd3 * log(E[i])^2
            );
            exchange = bm1 / E[i] + 1 + b1 * E[i] + b2 * E[i]^2;
            y[i] = y[i] + FD * exchange * pb * (E[i] + me) * (Q - E[i]) * sqrt(
                (Q - E[i])^ 2 - m_nu^2
            );
        }
    }

    return y;
}


vector Re187_bare(vector E){
    int N = num_elements(E);
    vector[N] y = rep_vector(0, N);

    real bm1 = 19.5,  b1 = -6.8e-6,  b2 = 3.05e-9;
    real fd1 = 3.01258188e02, fd2 = -4.98343890e-01, fd3 = 5.69632611e-04;
    real me = 510998.95;

    real pb;
    real FD;
    real exchange;

    for (i in 1:N) {
        pb = sqrt(E[i]^2 + 2 * E[i] * me);
        FD = exp(
            log(fd1) +
            fd2 * log(E[i]) +
            fd3 * log(E[i])^2
        );
        exchange = bm1 / E[i] + 1 + b1 * E[i] + b2 * E[i]^2;
        y[i] = y[i] + FD * exchange * pb * (E[i] + me);
    }

    return y;
}


vector gauss_exp_tail(vector E, real E0, real sigma, real lambda){
    int N = num_elements(E);
    vector[N] y = rep_vector(0, N);
    real sl = sigma*lambda;
    for (i in 1:N){
        y[i] = lambda/2 * exp((E[i]-E0)*lambda+sl^2 / 2)*erfc(((E[i]-E0)/sigma+sl)/sqrt(2));
    }
    return y;
}

vector gauss_plus_double_exp(vector E, real E0, real sigma, vector lambda, vector A_exp){
    int N = num_elements(E);
    vector[N] y = rep_vector(0, N);
    vector[2] sl = sigma*lambda;
    for (i in 1:N){
        real x = (E[i]-E0)/sigma;
        y[i] = A_exp[1] * lambda[1]/2 * exp((E[i]-E0)*lambda[1]+sl[1]^2 / 2)*erfc((x+sl[1])/sqrt(2));
        y[i] += A_exp[2] * lambda[2]/2 * exp((E[i]-E0)*lambda[2]+sl[2]^2 / 2)*erfc((x+sl[2])/sqrt(2));
        y[i] += (1-A_exp[1]-A_exp[2])*exp(-x^2 /2)/(sigma*sqrt(2*pi()));
    }
    return y;
}
