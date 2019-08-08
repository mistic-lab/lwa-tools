import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import inspect
import utils

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import plotTBN
import arrUtils


fc = arrUtils.frequency_dict[10]*10**6 # Hz
fs = 100000 # Hz

fft_size = 512 # bins

tend = 60*4 # s

f1 = 10000000 # 10MHz

# f1s = [f1, f1+.1, f1+.01, f1+.001]
f1s = [f1]
for f in f1s:
    simple_f = f-fc

    # Make the signal
    x = utils.make_complex_sine(simple_f, fs, tend, 0)
    fi = arrUtils.get_frequency_offset(fc, f, fft_size)
    period = arrUtils.get_period_from_frequency(fi, scale='m')
    # Plot the signal
    plotTBN.magnitude_of_timeseries(x, fs=fs, title='Frequency offset= '+str(round(fi, 4))+' Hz ('+str(round(period, 2))+' min)')
    # Plot the FFT of the input signal
    plotTBN.fft_full_length(x[:512], Fc=fc, title='FFT of input signals first ' + str(fft_size) + ' points')
    # Select a single bin of the fft
    bin_vector = arrUtils.get_vector_of_frequency_bin_from_time_series(x,Fc=fc,F1=f1)
    # Plot the magnitude of that bin
    # plotTBN.magnitude_of_timeseries(bin_vector,mode=None, title='Mag. of frequency bin | real | imag')
    # plotTBN.magnitude_of_timeseries(bin_vector,mode='lin', title='Mag of frequency bin | abs(real) | abs(imag)')
    # plotTBN.magnitude_of_timeseries(bin_vector,mode='log', title='Mag of frequency bin | 10log(abs(real)) | 10log(abs(imag))')
    # Plot the phase of the bin
    plotTBN.phase_of_timeseries(bin_vector, title='Phase of frequency bin')




