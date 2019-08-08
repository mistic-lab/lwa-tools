# IQ stream -> PFB -> DFT -> ACM

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys
import utils
import math

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import plotTBN
import arrUtils


fft_size = 512 # bins
overlap = 256 # bins


## Make IQ Streams

fc = arrUtils.frequency_dict[10]*10**6 # Hz
fs = 100e3 # Hz
f = 10e6
# tend = 60*4 # s
tend = 30 # s
simple_f = f-fc
x1 = utils.make_complex_sine(simple_f, fs, tend, 0)
x2 = utils.make_complex_sine(simple_f, fs, tend, 0)

#Plot inputs
# plotTBN.magnitude_of_timeseries(x1, fs,title='Magnitude of single antenna')
# plotTBN.phase_of_timeseries(x1, fs, title='Phase of single antenna')
# plotTBN.fft_full_length(x1, Fc=fc, title='FFT of single antenna')


## Run each stream through a PFB



## Run each stream through a DFT



## ACM those bad boys


#!/usr/bin/env python3
#
# Program to produce the power spectra from complex time series.
#
# Stephen Harrison
# NRC Herzberg
#
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="File to parse.")
parser.add_argument("-n", "--nfft", type=int, help="Number of FFT channels.")
parser.add_argument("-m", "--navg", type=int, help="Number of FFTs to average.")
parser.add_argument("-o", "--output", help="Output file name.")
args = parser.parse_args()
sz = args.nfft*args.navg
with open(args.filename, 'rb') as fi:
    with open(args.output, 'wb') as fo:
        x = np.fromfile(fi, dtype=np.int16, count=sz*2).astype(np.float32).view(np.complex64)
        while len(x) == sz:
            accum = np.zeros(args.nfft)
            for i in np.arange(0,sz,args.nfft):
                fft = np.fft.fftshift(np.fft.fft(x[i:i+args.nfft]*np.hanning(args.nfft)))
                accum += np.real(fft*np.conjugate(fft))
            accum /= float(args.nfft)
            accum.astype(np.float32).tofile(fo)
            print('.', end='', flush=True)
            x = np.fromfile(fi, dtype=np.int16, count=sz*2).astype(np.float32).view(np.complex64)
print()