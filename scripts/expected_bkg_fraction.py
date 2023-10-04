from baynes.toyMC import SpectraSampler
from baynes import HoSpectrum
from scipy import stats
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(A_Ho, R_t, bkg, ROI, m_nu, Q):
    f_pu = (stats.poisson.pmf(2, R_t * A_Ho)) / (stats.poisson.pmf(1, R_t * A_Ho))
    s = SpectraSampler({'$^{163}Ho$': [HoSpectrum, [m_nu, Q], A_Ho]}, f_pileup=f_pu, flat_bkg=bkg, ROI=ROI)

    print('Total fraction of pileup events: ', np.round(f_pu, 8))
    print('Fraction of pileup events in ROI: ', np.round(s.weights_in_ROI[1], 8))
    print('Fraction of background events in ROI: ', np.round(s.weights_in_ROI[2], 8))
    s.plot_spectrum()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the expected backround and pileup fraction in the ROI.")
    parser.add_argument("--A_Ho", type=float, default=1, help="Activity in the pixel")
    parser.add_argument("--R_t", type=float, default=3e-6, help="Time resolution")
    parser.add_argument("--bkg", type=float, default=1e-4, help="Rate of flat background")
    parser.add_argument("--ROI", type=float, nargs=2, default=[2650, 2900], help="Energy range of interest as list of float")
    parser.add_argument("--m_nu", type=float, default=0., help="Neutrino mass")
    parser.add_argument("--Q", type=float, default=2833., help="Spectrum Q-value")
    args = parser.parse_args()
    main(args.A_Ho, args.R_t, args.bkg, args.ROI, args.m_nu, args.Q)