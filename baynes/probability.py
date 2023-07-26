import numpy as np
import numbers

def lorentzian(E, E0, gamma):
    gamma_2 = gamma / 2
    return 1 / np.pi * gamma_2 / ((E - E0) ** 2 + gamma_2**2)

def lorentzian_asymm(E, E0, gamma, asymm):
    gamma_2 = gamma / 2
    gamma_L = (1-asymm) * gamma_2
    gamma_R = (1+asymm) * gamma_2
    if isinstance(E, list):
        E = np.array(E)
    if isinstance(E,np.ndarray) is False:
        if E < E0:
            y = 1/np.pi * (gamma_L**2 )/ ((E - E0)**2 + gamma_L**2)

        else:
            y = 1/np.pi * (gamma_R**2 )/ ((E - E0)**2 + gamma_R**2)
    else:
        y = np.zeros(len(E))
        pos_dx = np.where(E >= E0)[0]
        pos_sx = np.where(E < E0)[0]
        SS = E[pos_sx]
        DD = E[pos_dx]
        y[pos_sx] = gamma_L**2/((SS - E0)**2 + gamma_L**2)
        y[pos_dx] = gamma_R**2/((DD - E0)**2 + gamma_R**2)
    norm1 = 1./(np.pi*gamma_L*2)
    norm2 = 1./(np.pi*gamma_R*2)
    y = y *(norm1 + norm2)
    return y

def HoSpectrum(
    E,
    m_nu,
    Q_H=2838,
    E_H=[2047, 1842, 414.2, 333.5, 49.9, 26.3],
    gamma_H=[13.2, 6.0, 5.4, 5.3, 3.0, 3.0],
    i_H=[1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015],
):
    if isinstance(E, numbers.Number):
        spectrum = np.array([0.0])
    else:
        spectrum = np.zeros(len(E))
    for i in range(len(E_H)):
        spectrum += i_H[i] * lorentzian(E, E_H[i], gamma_H[i])
    return (
        np.clip((Q_H - E), 0, None)
        * np.sqrt(np.clip((Q_H - E) ** 2 - m_nu**2, 0, None))
        * spectrum
    )


def hdi(samples, prob=0.95):
    n = len(samples)
    sorted_samples = np.sort(samples)
    interval_len = int(np.floor(prob * n))
    n_intervals = n - interval_len
    interval_widths = np.subtract(
        sorted_samples[interval_len:], sorted_samples[:n_intervals]
    )

    min_idx = np.argmin(interval_widths)
    hdi_interval = sorted_samples[[min_idx, min_idx + interval_len]]

    return hdi_interval

def coverage(true_par, posteriors, prob=0.95):
    N = len(posteriors)
    Cv = 0.
    for post in posteriors:
        if isinstance(prob, numbers.Number):
            lo, hi = hdi(post, prob=prob)
        else:
            lo, hi = np.percentile(post, prob)
        if true_par==0:
            true_par = min(post)
        if lo < true_par and true_par < hi:
            Cv = Cv+1
    Cv = Cv/N
    C_err = np.sqrt(Cv*(1-Cv)/N)
    return Cv, C_err