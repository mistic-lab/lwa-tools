##########
# Author: Nicholas Bruce
# Date: July 10 2019
#
# Functions to plot LWASV data. LSL not required.
#
##########

from sys import argv
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm


def spectrogram(arr, Fc, Fs=100000, NFFT=2048, noverlap=1024, mode=None):
    """Wrapper for the specgram fn from matplotlib. Defaults to TBN filter 7.

    Parameters
    ----------
    arr : array
                numpy array of time series data from a single antenna
    Fc : float
                center frequency
    Fs : float
                sampling frequency in Hz (default: 100000)
    NFFT : int
                FFT size (default: 2048)
    noverlap : int
                FFT overlap size (default: 1024)
    mode : string
                options are 'psd' (default), 'magnitude', 'phase' (wrapped) 
                'angle' (unwrapped)

    Returns
    -------
    numpy array
        array of size (NFFT, length of spectrogram)
    """


    if mode == 'magnitude':
        cmap = cm.Reds
        cbarlabel = 'Magnitude [?]'
    elif mode in ['angle', 'phase']:
        cmap = cm.seismic
        cbarlabel = 'Phase [rads]'
    else:
        cbarlabel = ''
        cmap = None

    fig, ax = plt.subplots()
    spec, freqs, times, im = ax.specgram(arr, Fs=Fs, NFFT=NFFT, noverlap=noverlap, mode=mode, Fc=Fc, cmap=cmap)
    ax.set_title(mode)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    fig.colorbar(im).set_label(cbarlabel)
    fig.show()

    return spec


def histogram_full_BW(arr):
    """Wrapper for the hist fn from matplotlib. Sets bin edges based on quantization level of TBN data. Parameters are set for a full 100kHz BW time series.

    Parameters
    ----------
    arr : array
                numpy array of complex time series data from a single antenna
    """

    # approx 150 levels
    binedges = np.arange(-80.5,80.5, 1)

    #two horizontal sub plots
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Histogram of whole timeseries (full 100kHz BW)')                                                                                                                                                                         
    ax1.hist(np.real(arr), bins=binedges, label='real') 
    ax1.set_title('Real')  
    ax2.hist(np.imag(arr), bins=binedges, label='imag')
    ax2.set_title('Imaginary')
    plt.show()


def histogram_single_bin(arr):
    """Wrapper for the hist fn from matplotlib. Sets bin edges based on quantization level of TBN data. Parameters are set for a single bin of FFTd data.

    Parameters
    ----------
    arr : array
                numpy array of complex time series of FFT data from a single antenna
    """
    # dunno how many levels
    binedges = 512

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True) 
    f.suptitle('Histogram of full timeseries of FFT bin (195 Hz BW)')                                                                                                                                                                        
    ax1.hist(np.real(arr), bins=binedges, label='real')  
    ax1.set_title('Real')   
    ax2.hist(np.imag(arr), bins=binedges, label='imag') 
    ax2.set_title('Imaginary')
    plt.show()


def magnitude_of_timeseries(arr, fs=-1, title='Magnitude', mode=None):
    """Plots magnitude against time timeseries.

    Parameters
    ----------
    arr : array
                numpy array of complex time series data from a single antenna
    fs : int
                sampling rate of arr. If not provided, it plots arr against 'samples'
    title : string
                title to put in the plot
    mode : string
                One of 'log' (10.*np.log10(abs(arr))), 'lin' (abs(arr)), None (default, arr)
    """

    if fs == -1:
        t = range(len(arr))
        xlabel='Times (samples)'
    else:
        t = np.linspace(0,len(arr)/fs,len(arr))
        xlabel='Times (s)'

    if mode in [None, 'log', 'lin']:
        series_I = np.real(arr)
        series_Q = np.imag(arr)
    if mode in ['log', 'lin']:
        series_I = np.abs(series_I)
        series_Q = np.abs(series_Q)
    if mode == 'log':
        series_I = 10.*np.log10(series_I)
        series_Q = 10.*np.log10(series_Q)
    if mode not in [None, 'log', 'lin']:
        raise Exception("mode must be one of:  None, 'log', 'lin'")


    #Magnitude of full timeseries
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle(title)
    # f.suptitle('Magnitude of whole timeseries (full 100kHz BW)')                                                                                                                                                                         
    # f.suptitle('Magnitude of full timeseries of FFT bin (195 Hz BW)')                                                                                                                                                                         
    ax1.plot(t, series_I, label='real') 
    ax1.set_title('Real')  
    ax2.plot(t, series_Q, label='real') 
    ax2.set_title('Imaginary')
    ax2.set_xlabel(xlabel)
    plt.show()


def phase_of_timeseries(arr, fs=-1, title='Phase'):
    """Plots both phase and unwrapped phase against timeseries.

    Parameters
    ----------
    arr : array
                numpy array of complex time series data from a single antenna
    fs : int
                sampling rate of arr. If not provided, it plots arr against 'samples'
    title : string
            title to put in the plot

    """

    if fs == -1:
        t = range(len(arr))
        xlabel='Times (samples)'
    else:
        t = np.linspace(0,len(arr)/fs,len(arr))
        xlabel='Times (s)'

    #Magnitude of full timeseries
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle(title)
    ax1.plot(t, np.angle(arr), label='phase') 
    ax1.set_title('Wrapped')  
    ax2.plot(t, np.unwrap(np.angle(arr)), label='unwrapped phase') 
    ax2.set_title('Unwrapped')
    ax2.set_xlabel(xlabel)
    plt.show()

def fft_full_length(arr, Fc, Fs=100000, title='FFT'):
    """Plots an FFT of whatever you shove in.

    Parameters
    ----------
    arr : array
                numpy array of complex time series data from a single antenna
    Fc : float
                center frequency of time series data
    Fs : int
                sampling rate (default: 100000 Hz as per TBN filter code 7)
    title : string
            title to put in the plot

    """

    freqs = np.fft.fftshift(np.fft.fftfreq(len(arr), d=1.0/Fs)) + Fc

    fftData = 10.*np.log10(abs(np.fft.fftshift(np.fft.fft(arr)) ))

    f, ax = plt.subplots()
    f.suptitle(title)
    ax.plot(freqs, fftData)
    ax.set_ylabel('dB')
    ax.set_xlabel('Hz')
    plt.show()
