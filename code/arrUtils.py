##########
# Author: Nicholas Bruce
# Date: July 18 2019
#
# Functions to play with arrays of time/frequency data. LSL not required.
#
##########

import scipy.io.wavfile as wavf
import numpy as np
import math

def writeToWav(filename, arr, fs=100000):
    """Writes complex array to a wav file

    Parameters
    ----------
    filename : string
                name of file to be created (probably make it end in .wav)
    arr : complex array
                time series
    fs : int
                sampling frequency in Hz (default: 100000)
    """

    v = arr.view(np.float32)
    w = v.reshape(arr.shape + (2,))
    wavf.write(filename, fs, w)


def get_frequency_offset(fc, f1, fft_size, fs=100000, show_details=False):
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
        print('bin_size: {}'.format(bin_size))
        print('hz_between_freqs: {}'.format(hz_between_freqs))
        print('bins_between_freqs: {}'.format(bins_between_freqs))
        print('bins_from_edge_of_fc: {}'.format(bins_from_edge_of_fc))
        print('bin_amount_from_edge_of_f1: {}'.format(bin_amount_from_edge_of_f1))
        print('hz_from_edge_of_f1: {}'.format(hz_from_edge_of_f1))

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


def get_vector_of_frequency_bin_from_time_series(timeseries, Fc, F1, Fs=100000, fft_size=512, overlap_size=256):
    """Takes the FFT of a timeseries, extracts the content of a single bin, shifts to an overlapped FFT, repeasts.

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
    numpy array
        complex vector containing the data from a single FFT bin
    """

    bin_containing_f1 = get_frequency_bin(fc=Fc, f1=F1, fft_size=fft_size)

    # Using a list because append is handy, and I don't want to math how long it's going to be
    output_data = []

    start = 0
    while start + fft_size < len(timeseries):
        # grab a an fft window and fft it
        snippet = timeseries[start:start+fft_size]
        fft=np.fft.fftshift(np.fft.fft(snippet))

        # take the single complex number from the bin of interest and store it
        pt = fft[bin_containing_f1]
        output_data.append(pt)

        start += overlap_size
        
    #numpify that bad boy
    output_data = np.array(output_data)

    return output_data

def get_period_from_frequency(f, scale='s'):
    """
    Title says it all.

    Parameters
    ----------
    fi : float
                frequency
    scale : string
                time scale to return period in. Options are 's' (default, seconds), 'm' (minutes), 'h' (hours)

    Returns
    -------
    float
        period in either mins, seconds, or hours
    """

    if scale not in ['s', 'm', 'h']:
        raise Exception("scale must be either 's', 'm' or 'h'")
    else:
        period = f**-1
    
    if scale in ['m', 'h']:
        period = period/60
    if scale in ['h']:
        period = period/60
    
    return period