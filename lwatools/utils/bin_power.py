import h5py
import argparse

import numpy as np
from scipy.signal import welch, hann

from lsl.reader.ldp import LWASVDataFile

def estimate_snr(data, fft_len, target_freq, fs, fc):
    '''
    Uses Welch's periodogram method to estimate signal and noise powers.
    '''

    # use a periodic hann window
    win = hann(fft_len, sym=False)

    # overlap is nperseg // 2 by default
    freqs, pxx = welch(x=data, fs=fs, nperseg=fft_len, return_onesided=False, window=win, detrend=False)

    # shift frequencies up from baseband
    freqs += fc

    target_bin = np.argmin([abs(target_freq - f) for f in freqs])

    print(f"using bin {target_bin}")
    
    # the window shape will show how the signal leaks into neighboring bins
    # compute the bin shape in the frequency domain
    fwin = np.abs(np.fft.fft(win))
    # center it where we think the signal is
    fwin = np.roll(fwin, target_bin)
    # the signal leaks where the window magnitude is large
    sig_idx = np.abs(fwin) > 1
    nosig_idx = np.invert(sig_idx)
    
    noise_density_est = pxx[nosig_idx].mean()
    noise_power_est = noise_density_est * fs

    sig_bins = pxx[sig_idx]
    sig_power_est = (np.sum(sig_bins) - len(sig_bins) * noise_density_est) * fs / fft_len

    snr_est = sig_power_est / noise_power_est

    return snr_est
