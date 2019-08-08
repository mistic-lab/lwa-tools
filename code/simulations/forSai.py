## Simulate sampling some frequency at some rate
# Author: Nicholas Bruce (nsbruce@uvic.ca)
# Date: Aug 2019


################################### IMPORTS ###################################

import matplotlib.pyplot as plt
import numpy as np
import math




################################### USEFUL FUNCTIONS ###################################

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

def get_frequency_bin(fc, f1, fft_size, fs=100000, show_details=False):
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




################################### MAIN ###################################

def main():

    ## Parameters from LWA
    fc = 9974999.9987 # Hz
    fs = 100000 # Hz
    f1 = 10000000 # 10MHz

    ## Chosen simulation parameters
    fft_size = 512 # bins
    tend = 60*4 # seconds


    ## Bring fc down to 0
    simple_f = f1-fc

    # Make the complex signal
    t_arr = np.linspace(0,tend, tend*fs) # time in s
    omega = simple_f * 2 * np.pi
    phi = 0 # rads
    x = np.exp(1j*(omega * t_arr + phi)) # This is our sampled sampled signal

    # Plot the signal
    series_I = np.real(x)
    series_Q = np.imag(x)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Magnitude of input signal')
    ax1.plot(series_I, label='real') 
    ax1.set_title('Real')  
    ax2.plot(series_Q, label='real') 
    ax2.set_title('Imaginary')
    ax2.set_xlabel('Samples')
    plt.show()



    # Plot the FFT of the input signal
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x[:1000]), d=1.0/fs)) + fc
    fftData = 10.*np.log10(abs(np.fft.fftshift(np.fft.fft(x[:1000])) )) # only plotting the first 1000 samples

    f, ax = plt.subplots()
    f.suptitle('FFT of first 1000 pts')
    ax.plot(freqs, fftData)
    ax.set_ylabel('dB')
    ax.set_xlabel('Hz')
    plt.show()


    # Select a single bin of the fft
    bin_vector = get_vector_of_frequency_bin_from_time_series(x,Fc=fc,F1=f1)

    # Plot the magnitude of that bin
    # plotTBN.magnitude_of_timeseries(bin_vector,mode=None, title='Mag. of frequency bin | real | imag')
    # plotTBN.magnitude_of_timeseries(bin_vector,mode='lin', title='Mag of frequency bin | abs(real) | abs(imag)')
    # plotTBN.magnitude_of_timeseries(bin_vector,mode='log', title='Mag of frequency bin | 10log(abs(real)) | 10log(abs(imag))')

    # Plot the phase of the bin
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Phase of FFT bin')
    ax1.plot(np.angle(x), label='phase') 
    ax1.set_title('Wrapped')  
    ax2.plot(np.unwrap(np.angle(x)), label='unwrapped phase') 
    ax2.set_title('Unwrapped')
    ax2.set_xlabel('Samples')
    plt.show()


if __name__ == "__main__":
    main()