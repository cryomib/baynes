"""Statistics-related functions and energy spectra distributions."""
import numbers
from math import erf

import numpy as np
from numba import njit


def lorentzian(E, E0, gamma):
    """
    Calculate the value of the Lorentzian function at a given energy.

    Parameters:
        E (float or array_like): The energy value(s) where the Lorentzian is evaluated.
        E0 (float): The center of the Lorentzian curve.
        gamma (float): The full width at half maximum (FWHM) of the Lorentzian curve.

    Returns:
        float or ndarray: The value of the Lorentzian function at the given energy value(s).
    """
    E = np.asarray(E)
    gamma_2 = gamma / 2
    return 1 / np.pi * gamma_2 / ((E - E0) ** 2 + gamma_2**2)


def lorentzian_asymm(E, E0, gamma, asymm):
    """
    Calculate the value of the asymmetric Lorentzian function at a given energy.

    Parameters:
        E (float or array_like): The energy value(s) where the asymmetric Lorentzian is evaluated.
        E0 (float): The center of the Lorentzian curve.
        gamma (float): The full width at half maximum (FWHM) of the Lorentzian curve.
        asymm (float): The asymmetry parameter that shifts the FWHM.

    Returns:
        float or ndarray: The value of the asymmetric Lorentzian function at the given energy value(s).
    """
    gamma_2 = gamma / 2
    gamma_L = (1 - asymm) * gamma_2
    gamma_R = (1 + asymm) * gamma_2

    if not isinstance(E, np.ndarray):
        E = np.array([E])

    y = np.where(
        E < E0,
        gamma_L**2 / ((E - E0) ** 2 + gamma_L**2),
        gamma_R**2 / ((E - E0) ** 2 + gamma_R**2),
    )

    norm1 = 1.0 / (np.pi * gamma_L * 2)
    norm2 = 1.0 / (np.pi * gamma_R * 2)
    y = y * (norm1 + norm2)

    if len(y) == 1:
        return float(y[0])
    return y


def HoSpectrum(
    E,
    m_nu,
    Q_H=2833,
    E_H=[2047, 1842, 414.2, 333.5, 49.9, 26.3],
    gamma_H=[13.2, 6.0, 5.4, 5.3, 3.0, 3.0],
    i_H=[1, 0.0526, 0.2329, 0.0119, 0.0345, 0.0015],
):
    """
    Calculate the first-order 130Ho electronic capture decay spectrum.

    Parameters:
        E (float or array_like): The energy value(s) where the spectrum is evaluated.
        m_nu (float): The neutrino mass.
        Q_H (float, optional): The decay energy. Default is 2838 keV.
        E_H (list of float, optional): List of peak energies in the spectrum. Default values are provided.
        gamma_H (list of float, optional): List of peak widths (FWHM) in the spectrum. Default values are provided.
        i_H (list of float, optional): List of peak intensities in the spectrum. Default values are provided.

    Returns:
        float or ndarray: The value of the first-order 130Ho electronic capture decay spectrum at the given energy value(s).
    """
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


@njit()
def re_spectrum(E, m_nu, Q=2465, bm1=19.5, b1=-6.8e-6, b2=3.05e-9):
    """Compute the 187Re spectrum with shape factor corrections."""
    E = np.asarray(E)
    N = len(E)
    y = np.zeros(N)
    me = 510998
    Gf = 1.1663787e-23
    Vud = 0.97373
    gsq = (Gf * Vud) ** 2
    FD_pars = [3.01258188e02, -4.98343890e-01, 5.69632611e-04]
    for i in range(N):
        if Q - E[i] >= m_nu:
            pb = np.sqrt(E[i] ** 2 + 2 * E[i] * me)
            FD = np.exp(
                np.log(FD_pars[0])
                + FD_pars[1] * np.log(E[i])
                + FD_pars[2] * (np.log(E[i])) ** 2
            )
            exchange = bm1 / E[i] + 1 + b1 * E[i] + b2 * E[i] ** 2
            y[i] = y[i] + FD * exchange * pb * (E[i] + me) * (Q - E[i]) * np.sqrt(
                (Q - E[i]) ** 2 - m_nu**2
            )
    return y


@njit()
def ptolemy(E, coeffs, m_nu, Q=18589.8):
    """Compute the b-decay spectrum of tritium in graphene."""
    E = np.asarray(E)
    N = len(E)
    y = np.zeros(N)
    me = 510998.95
    mhe3 = 931.49410242e6 * 3.0160293
    lambda_val = 4.21e-5
    eps0 = 5.76
    Gf = 1.1663787e-23
    Vud = 0.97373
    gsq = (Gf * Vud) ** 2 * (1 + (1.25 * 1.65) ** 2)
    NH3 = 1 / 1.66054e-24 / 3.01604928

    const_discrspe = NH3 * gsq * lambda_val**2 / (4 * np.pi**3 * 6.582119e-16)
    const_contspe = NH3 * gsq * lambda_val / (2 * np.pi ** (7.0 / 2.0) * 6.582119e-16)

    i_disc = lambda_val / (2 * np.pi**3)
    pb = np.sqrt(E**2 + 2 * E * me)

    for k in range(64):
        Qn = Q - coeffs[k][0]
        pn = coeffs[k][1]
        an = coeffs[k][2]
        bn = coeffs[k][3]
        cn = coeffs[k][4]
        for j in range(N):
            E_nu = Qn - E[j]
            if E_nu >= m_nu:
                xb = (pb[j] - pn) / pn
                y[j] += (
                    const_discrspe
                    * pb[j]
                    * (E[j] + me)
                    * np.sqrt(E_nu**2 - m_nu**2)
                    * E_nu
                    * (an + xb * lambda_val * pn * (2 * bn - lambda_val * pn * cn))
                )

    for j in range(N):
        QKE = Q - E[j] - eps0
        if QKE >= m_nu:
            b = -pb[j] + np.sqrt(2 * mhe3 * (QKE - m_nu))
            bl = b * lambda_val
            pl = pb[j] * lambda_val
            expbl = np.exp(-(bl**2))
            exppl = np.exp(-(pl**2))
            erfplbl = erf(pl) + erf(bl)

            pb2_2mhe3 = pb[j] ** 2 / (2 * mhe3)
            kinf2 = np.sqrt(1 - (m_nu / (4.0 + QKE - pb2_2mhe3)) ** 2)

            I0 = (
                np.sqrt(np.pi)
                / (2 * lambda_val)
                * pb[j]
                * (QKE - pb2_2mhe3) ** 2
                * erfplbl
            )
            I1 = (
                1
                / (2 * lambda_val**2)
                * (QKE - pb2_2mhe3)
                * (QKE - 5.0 * pb2_2mhe3)
                * (exppl - expbl)
            )
            I2 = (
                1
                / (4 * lambda_val**3)
                * ((5.0 * pb2_2mhe3 - 3.0 * QKE) * pb[j] / mhe3)
                * (-2 * pl * exppl - 2 * bl * expbl + np.sqrt(np.pi) * erfplbl)
            )
            I3 = (
                1
                / (2 * lambda_val**4)
                * ((5.0 * pb2_2mhe3 - QKE) / mhe3)
                * (exppl * (1 + pl**2) - expbl * (1 + bl**2))
            )
            I4 = (
                5
                * pb[j]
                / (32 * mhe3**2 * lambda_val**5)
                * (
                    -expbl * (6 * bl + 4 * bl**3)
                    - exppl * (6.0 * pl + 4 * pl**3)
                    + 3 * np.sqrt(np.pi) * erfplbl
                )
            )
            I5 = (
                1
                / (8 * mhe3**2 * lambda_val**6)
                * (
                    exppl * (2 + 2 * pl**2 + pl**4)
                    - expbl * (2.0 + 2.0 * bl**2 + bl**4)
                )
            )
            y[j] += const_contspe * (E[j] + me) * kinf2 * (I0 + I1 + I2 + I3 + I4 + I5)

    return y


@njit()
def allowed_beta(E, m_nu, Q=18589.6):
    """Compute the b-decay spectrum of atomic tritium."""
    E = np.asarray(E)
    N = len(E)
    y = np.zeros(N)
    me = 510998.95
    # alpha = 1/137
    # eta = Z* alpha * E/p
    # F = 2*np.pi/(1-np.exp(-2*np.pi*eta))
    Gf = 1.1663787e-23
    Vud = 0.97373
    NH3 = 1 / 1.66054e-24 / 3.01604928

    const = NH3 * (Gf * Vud) ** 2 * (1 + 3 * (1.2646**2))
    for i in range(N):
        if (Q - E[i]) > m_nu:
            p = np.sqrt(E[i] ** 2 + 2 * E[i] * me)
            if E[i] == 0:
                y[i] = 0
            else:
                beta = p / (E[i] + me)
                eta = 2.0 * 0.04585061813815046 / beta
                F = eta * (1.002037 - 0.001427 * beta) / (1 - np.exp(-eta))
                y[i] = (
                    F
                    * p
                    * (E[i] + me)
                    * (Q - E[i])
                    * np.sqrt((Q - E[i]) ** 2 - m_nu**2)
                )
    return y * const / (2 * np.pi**3 * 6.582119e-16)


def hdi(samples, prob=0.95):
    """
    Calculate the Highest Density Interval (HDI) for a given sample distribution.

    The HDI is an interval containing the specified probability mass of the distribution
    that has the highest density of probability.

    Parameters:
        samples (array_like): The sample distribution from which to calculate the HDI.
        prob (float, optional): The probability mass contained in the HDI (0 to 1). Default is 0.95.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the HDI.
    """
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
    """
    Calculate the coverage probability and its error for a set of posteriors.

    The coverage probability (Cv) measures the proportion of posteriors that include the true parameter value.
    It provides an estimate of the accuracy of the parameter estimation.

    Parameters:
        true_par (float): The true value of the parameter being estimated.
        posteriors (list of array_like): List of posterior distributions to calculate the coverage from.
        prob (float or array_like, optional): The probability mass or percentiles (0 to 100) for HDI calculation.
                                              Default is 0.95.

    Returns:
        tuple: A tuple containing the coverage probability (Cv) and its error (C_err).
    """
    N = len(posteriors)
    Cv = 0.0
    for post in posteriors:
        if isinstance(prob, numbers.Number):
            lo, hi = hdi(post, prob=prob)
        else:
            lo, hi = np.percentile(post, prob)
        if true_par == 0:
            true_par = min(post)
        if lo < true_par and true_par < hi:
            Cv += 1
    Cv /= N
    C_err = np.sqrt(Cv * (1 - Cv) / N)
    return Cv, C_err
