
import math


def get_frequency_offset_from_bin_center(fc, f1, fft_size, fs=100000, show_details=False):
    """Get's the frequency offset from the bin center for a signal

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
    float
        Hz from the center of the bin containing f1, to f1.
    """
    
    #   |<-->|      |    fc   |      |  f1   | [Hz]
    bin_size = fs/fft_size

    #   |    |      |    fc<--|------|->f1   | [Hz]
    hz_between_freqs = abs(f1-fc)

    #   |    |      |    fc<--|------|->f1   | [bins]
    bins_between_freqs = hz_between_freqs/bin_size

    #   |    |      |    fc   |<-----|->f1   | [bins]
    bins_from_edge_of_fc = bins_between_freqs-0.5

    #   |    |      |    fc   |      |<>f1   | [bins]
    bin_amount_from_edge_of_f1 = bins_from_edge_of_fc - math.floor(bins_from_edge_of_fc)

    #   |    |      |    fc   |      |<>f1   | [Hz]
    hz_from_edge_of_f1 = bin_size*bin_amount_from_edge_of_f1

    # |     f1<-------->center              | [Hz]
    hz_from_center_of_bin_containing_f1 = abs(hz_from_edge_of_f1-(bin_size/2))

    if show_details == True:
        print('| bin_size: {}'.format(bin_size))
        print('| hz_between_freqs: {}'.format(hz_between_freqs))
        print('| bins_between_freqs: {}'.format(bins_between_freqs))
        print('| bins_from_edge_of_fc: {}'.format(bins_from_edge_of_fc))
        print('| bin_amount_from_edge_of_f1: {}'.format(bin_amount_from_edge_of_f1))
        print('| hz_from_edge_of_f1: {}'.format(hz_from_edge_of_f1))

    return hz_from_center_of_bin_containing_f1


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
    
    return int(frequency_bin_of_f1)


