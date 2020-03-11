"""
Script that runs all of the h5 PFBs in a file into an ACM.
"""

import os
import numpy as np
import argparse
from datetime import datetime
import h5py
import time
import math
from arrUtils import get_frequency_bin


def __build_singleAnt_ACM__(foACM, fiPol, tIdx, antIdx, fft_size):
    for k in range(fft_size):
        x_k = np.array(fiPol[:,tIdx,k])
        x_k_h = x_k[antIdx].conj().T
        result = x_k * x_k_h
        foACM[:, tIdx, k] = result

def __build_singleFreq_ACM__(foACM, fiPol, tIdx, fBin):
    x_k = np.array(fiPol[:,tIdx,fBin])
    x_k = x_k.reshape(1,-1) # Cast to column vector
    x_k_h = x_k.conj().T
    result = x_k * x_k_h
    foACM[:, :, tIdx] = result

def __build_singleFreq_singleAnt_ACM__(foACM, fiPol, tIdx, antIdx, fBin):
    x_k = np.array(fiPol[:,tIdx,fBin])
    x_k_h = x_k[antIdx].conj().T
    result = x_k * x_k_h
    foACM[:, tIdx] = result # there was a tIdx, but I think it was a boooboo

def __build_full_ACM__(foACM, fiPol, tIdx, fft_size):
    for k in range(fft_size):
        x_k = np.array(fiPol[:,tIdx,k])
        x_k = x_k.reshape(1,-1) # Cast to column vector
        x_k_h = x_k.conj().T
        result = x_k * x_k_h
        foACM[:, :, tIdx, k] = result

def __build_singleFreq_dualPol_ACM__(foACM, fiPol0, fiPol1, tIdx, fBin):
    x_k = fiPol0[:,tIdx,fBin]
    x_k = x_k.reshape(1,-1) # Cast to column vector
    x_k_h = fiPol1[:,tIdx,fBin]
    x_k_h = x_k_h.reshape(1,-1) # Cast to column vector
    x_k_h = x_k_h.conj().T
    result = x_k * x_k_h
    foACM[:, :, tIdx] = result

def __build_singleFreq_singleAnt_dualPol_ACM__(foACM, fiPol0, fiPol1, tIdx, antIdx, fBin):
    x_k = np.array(fiPol0[:,tIdx,fBin])
    x_k_h = np.array(fiPol1[:,tIdx,fBin])
    x_k_h = x_k_h[antIdx].conj().T
    result = x_k * x_k_h
    foACM[:, tIdx] = result


def main(args):

    if args.freq == -100.0:
        args.freq = False
    if args.ant == -100:
        args.ant = False

    print("\nCreating filenames and checking input file extension")
    input_file = args.input


    ext = os.path.splitext(input_file)[-1].lower()
    if ext not in ['.h5', '.hdf5']:
        raise Exception("Extension should be .h5 or .hdf5, instead it is {}".format(str(ext)))
    else:
        input_filename = os.path.split(os.path.splitext(input_file)[-2])[-1][:-8]
        output_file = input_filename + '_ACM.hdf5'
        print("-| Input file is {}".format(input_file))
        print("-| Output file is {}".format(output_file))


    with h5py.File(args.input, 'r') as fi:

        # args.pol = int(args.pol)
        print("args.pol = {}".format(args.pol))

        print("-| Checking polarizations")
        if 0 in args.pol:
            fiPol0 = fi['pol0']
            print("--| Building pol0")
        if 1 in args.pol:
            fiPol1 = fi['pol1']
            print("--| Building pol1")

        with h5py.File(output_file,'w') as fo:

            print("-| Copying attributes")
            for key, value in fi.attrs.items():
                fo.attrs[key] = value
                print("--| key: {}  | value: {}".format(key, value))
            
            print("-| Checking for freq/antenna parameters")
            if args.freq:
                fBin = get_frequency_bin(fo.attrs['freq1'], args.freq, fo.attrs['nFFT'],fs=fo.attrs['sampleRate'])
                print("--| {} Hz is in bin {}".format(args.freq, fBin))
                print("--| Adding fBin to attributes")
                fo.attrs['fBin'] = fBin
            else:
                print("--| Building ACM for all available bins")
            if args.ant:
                idAnt = args.ant
                print("--| Only building relative to id {}".format(idAnt))
                print("--| Adding antID to attributes")
                fo.attrs['antID'] = idAnt

            else:
                print("--| Building ACM for all available indexes") 


            print("-| Copying frequency and time arrays")
            fo.create_dataset('times', (len(fi['times']),), dtype="float64")[:] = fi['times']
            fo.create_dataset('freqs', (len(fi['freqs']),), dtype="float32")[:] = fi['freqs']


            # Creating output size
            tlen = len(fi['times'])

            if args.freq and args.ant:
                fft_size = 1
                output_shape = (fi['pol0'].shape[0], tlen)
            elif args.freq and not args.ant:
                fft_size = 1
                output_shape = (fi['pol0'].shape[0],fi['pol0'].shape[0],tlen)
            elif args.ant and not args.freq:
                fft_size = len(fi['freqs'])
                output_shape = (fi['pol0'].shape[0], tlen, fft_size)
            else:
                fft_size = len(fi['freqs'])
                output_shape = (fi['pol0'].shape[0], fi['pol0'].shape[0], tlen, fft_size)
            
            print("-| Output ACM shape is: {}".format(output_shape))

            # Create a subdataset for each polarization
            print("-| Creating datasets full of zeros")
            if 0 in args.pol:
                print("--| pol0")
                foPol0_ACM = fo.create_dataset("pol0", output_shape, dtype=np.complex64)
            if 1 in args.pol:
                print("--| pol1")
                foPol1_ACM = fo.create_dataset("pol1", output_shape, dtype=np.complex64)
            if 0 in args.pol and 1 in args.pol:
                print("--| pol01")
                foPol01_ACM = fo.create_dataset("pol01", output_shape, dtype=np.complex64)
                print("--| pol10")
                foPol10_ACM = fo.create_dataset("pol10", output_shape, dtype=np.complex64)

            if 0 in args.pol:
                for t in range(tlen):
                    print("t: {}/{}".format(t, tlen-1))
                    if args.freq and args.ant:
                        __build_singleFreq_singleAnt_ACM__(foPol0_ACM,fiPol0, t, idAnt,fBin)
                    elif args.freq and not args.ant:
                        __build_singleFreq_ACM__(foPol0_ACM, fiPol0,t,fBin)
                    elif args.ant and not args.freq:
                        __build_singleAnt_ACM__(foPol0_ACM,fiPol0, t, idAnt,fft_size)
                    else:
                        __build_full_ACM__(foPol0_ACM, fiPol0, t, fft_size)

            if 1 in args.pol:
                for t in range(tlen):
                    print("t: {}/{}".format(t, tlen-1))
                    if args.freq and args.ant:
                        __build_singleFreq_singleAnt_ACM__(foPol1_ACM,fiPol1, t, idAnt,fBin)
                    elif args.freq and not args.ant:
                        __build_singleFreq_ACM__(foPol1_ACM, fiPol1,t,fBin)
                    elif args.ant and not args.freq:
                        __build_singleAnt_ACM__(foPol1_ACM,fiPol1, t, idAnt,fft_size)
                    else:
                        __build_full_ACM__(foPol1_ACM, fiPol1, t, fft_size)

            if 0 in args.pol and 1 in args.pol:
                for t in range(tlen):
                    print("t: {}/{}".format(t, tlen-1))
                    if args.freq and args.ant:
                        __build_singleFreq_singleAnt_dualPol_ACM__(foPol1_ACM,fiPol0, fiPol1, t, idAnt,fBin)
                        __build_singleFreq_singleAnt_dualPol_ACM__(foPol1_ACM,fiPol1, fiPol0, t, idAnt,fBin)
                    elif args.freq and not args.ant:
                        __build_singleFreq_dualPol_ACM__(foPol01_ACM, fiPol0, fiPol1, t, fBin)
                        __build_singleFreq_dualPol_ACM__(foPol10_ACM, fiPol1, fiPol0, t, fBin)
                    # elif args.ant and not args.freq:
                    #     __build_singleAnt_ACM__(foPol1_ACM,fiPol1, t, idAnt,fft_size)
                    # else:
                    #     __build_full_ACM__(foPol1_ACM, fiPol1, t, fft_size)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates an ACM from integrated spectra.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('input', type=str,
                        help='input h5 file of integrated spectra')
    parser.add_argument('-f', '--freq', type=float, default = -100.0,
                        help='frequency to keep in Hz')
    parser.add_argument('-a', '--ant', type=int, default = -100,
                        help='antenna index to keep')
    parser.add_argument('-p', '--pol', type = int, action='append',
                        help='polarization(s) to build. Call flag as many times as you want to get multiple polarizations')
    args = parser.parse_args()
    main(args)