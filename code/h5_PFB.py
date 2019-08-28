"""
Script that runs all of the h5 time streams through a PFB and DFT.
"""

import os
import numpy as np
import argparse
from datetime import datetime
import h5py
import time
import math

def round_down(num, divisor):
    return int(num - (num%divisor))

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

############################# Setup output h5 file #############################

def main(args):

    print("\nCreating filenames and checking input file extension")
    input_file = args.input


    ext = os.path.splitext(input_file)[-1].lower()
    if ext not in ['.h5', '.hdf5']:
        raise Exception("Extension should be .h5 or .hdf5, instead it is {}".format(str(ext)))
    else:
        input_filename = os.path.split(os.path.splitext(input_file)[-2])[-1]
        output_file = input_filename + '_postPFB.hdf5'
        print("-| Input file is {}".format(input_file))
        print("-| Output file is {}".format(output_file))


    with h5py.File(args.input, 'r') as fi:

        fiParent = fi[input_filename]
        fiPol0 = fiParent['pol0']
        fiPol1 = fiParent['pol1']


        with h5py.File(output_file,'w') as fo:

            # Create a group to store everything in, and to attach attributes to
            # print("-| Creating parent group {}[{}]".format(output_file, input_filename))
            # foParent = fo.create_group(input_filename)

            # Add attributes to group
            print("-| Copying attributes across")
            for key, value in fiParent.attrs.items():
                fo.attrs[key] = value
                print("--| key: {}  | value: {}".format(key, value))
            print("-| Adding new attributes")
            fo.attrs['nFFT'] = args.nfft
            fo.attrs['nAVG'] = args.navg

            # Creating output size
            tsLen = fiPol0.shape[1]
            print("-| Input time series length is {} samples".format(tsLen))
            output_shape = (fiPol0.shape[0], int(round_down(tsLen,(args.nfft*args.navg))/(args.nfft*args.navg)), args.nfft)
            print("-| Number of output channels containing spectrograms is {}".format(output_shape[0]))
            print("-| Output spectrogram shape is {} frequency bins by {} time samples".format(output_shape[2], output_shape[1]))

            # Create a subdataset for each polarization
            print("-| Creating datasets full of zeros")
            foPol0 = fo.create_dataset("pol0", output_shape, dtype=np.complex64)#, compression='lzf')
            foPol1 = fo.create_dataset("pol1", output_shape, dtype=np.complex64)#, compression='lzf')

            times = np.linspace(0, output_shape[1]*args.nfft*args.navg/args.fs, output_shape[1], endpoint=False)+int(input_filename)
            freqs = np.linspace(-args.fs/2, args.fs/2, args.nfft, endpoint=False)+args.fc

            fo.create_dataset('times', (len(times),), dtype="float64")[:] = times
            fo.create_dataset("freqs", (len(freqs),), dtype="float32")[:] = freqs

            for i in range(len(fiPol0)):
                foPol0[i] = quick_pfb(fiPol0[i],args.nfft, args.navg)
                foPol1[i] = quick_pfb(fiPol1[i],args.nfft, args.navg)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates integrated complex spectra from time series.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-n', '--nfft', type=int, default=1000, 
                        help='fft size')
    parser.add_argument('-a', '--navg', type=int, default=10, 
                        help='number of ffts to average (PFB taps)')
    parser.add_argument('-s', '--fs', type=int, default=100e3, 
                        help='sampling frequency in Hz')
    parser.add_argument('-c', '--fc', type=float, default=5024999.967776239, 
                        help='sampling frequency in Hz')
    parser.add_argument('-i', '--input', type=str,
                        help='input h5 file of time series')
    args = parser.parse_args()
    main(args)