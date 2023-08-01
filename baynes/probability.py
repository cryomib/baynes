import numpy as np
import numbers

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
        E = np.array([E])  # Convert scalar input to 1-element ndarray

    y = np.where(E < E0, gamma_L**2 / ((E - E0)**2 + gamma_L**2),
                         gamma_R**2 / ((E - E0)**2 + gamma_R**2))

    norm1 = 1. / (np.pi * gamma_L * 2)
    norm2 = 1. / (np.pi * gamma_R * 2)
    y = y * (norm1 + norm2)

    if len(y) == 1:
        return float(y[0])  # Return float for scalar input
    return y

def HoSpectrum(
    E,
    m_nu,
    Q_H=2838,
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


import numpy as np
import numbers

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
