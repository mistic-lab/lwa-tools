import math
import numpy as np

############################# GET BIN NUMBER #############################
def get_frequency_bin(fc, f1, fft_size, fs=100000, show_details=False):
    """Gets the bin number for some frequency.

    Parameters
    ----------
    fc : float
                center frequency in Hz
    f1 : float
                signal of interest in Hz
    fft_size : int
                size of the FFT used
    fs : int
                sampling frequency in Hz (default: 100000)
    show_details: boolean
                sanity check that prints out each line (default: False)

    Returns
    -------
    int
        bin containing f1
    """

    center_bin = fft_size/2
    bin_size = fs/fft_size

    hz_between_freqs = abs(f1-fc)

    if hz_between_freqs < bin_size/2:
        return center_bin

    bins_between_freqs = hz_between_freqs/bin_size

    bins_from_edge_of_fc = math.ceil(bins_between_freqs-0.5)

    if f1 < fc:
        frequency_bin_of_f1 = center_bin - bins_from_edge_of_fc
    elif fc < f1:
        frequency_bin_of_f1 = center_bin + bins_from_edge_of_fc
    
    if show_details:
        print("fc bin = {}".format(center_bin))
        print("bin size = {} Hz".format(bin_size))
        print("|f1-fc| = {} Hz".format(hz_between_freqs))
        print("|f1-fc| = {} bins".format(bins_between_freqs))
        print("bins from edge of fc's bin = {}".format(bins_from_edge_of_fc))
        print("frequency bin of f1 = {}".format(int(frequency_bin_of_f1)))

    return int(frequency_bin_of_f1)


############################# PFB #############################
def quick_pfb(series, nfft, navg):
    """Runs a signal through a polyphase filter bank.

    Parameters
    ----------
    series : array
                time series to be run through the polyphase
    nfft : int
                fft size to be used
    navg : int
                number of ffts to integrate over

    Returns
    -------
    numpy array
        shape is (time_samples, nfft)
    """
    sz = nfft * navg
    spec = []
    for k in np.arange(0, len(series), sz):
        x = series[k:k+sz]
        accum = np.zeros(nfft, dtype=x.dtype)
        if len(x) == sz:
            for i in np.arange(0, sz, nfft):
                fft = np.fft.fftshift(np.fft.fft(x[i:i+nfft]*np.hanning(nfft)))
                accum += fft
            accum /= float(navg)
            spec.append(accum)

    # spec = np.rot90(np.array(spec),3)
    spec = np.array(spec)

    return spec


############################# ACM #############################
def quick_ACM(phys_chan_1, phys_chan_2):
    """Runs two waterfalls into an array covariance matrix.

    Parameters
    ----------
    phys_chan_1 : array
                MUST be of shape (tlen, fft_size)

    Returns
    -------
    numpy array
        shape is (2, 2, tlen, fft_size)
    """
    tlen = phys_chan_1.shape[0]
    fft_size = phys_chan_1.shape[1]

    ACM = np.zeros((2, 2, tlen, fft_size), dtype=phys_chan_1.dtype)
    
    for t in range(tlen):
        for k in range(fft_size):
            x_k = np.array((phys_chan_1[t,k],phys_chan_2[t,k]))
            x_k = x_k.reshape(1,-1) # Cast to column vector
            x_k_h = x_k.conj().T
            ACM[:, :, t, k] = x_k * x_k_h

    return ACM
