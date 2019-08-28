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

def main(args):

    print("\nCreating filenames and checking input file extension")
    input_file = args.input


    ext = os.path.splitext(input_file)[-1].lower()
    if ext not in ['.h5', '.hdf5']:
        raise Exception("Extension should be .h5 or .hdf5, instead it is {}".format(str(ext)))
    else:
        input_filename = os.path.split(os.path.splitext(input_file)[-2])[-1]
        output_file = input_filename + '_ACM.hdf5'
        print("-| Input file is {}".format(input_file))
        print("-| Output file is {}".format(output_file))


    with h5py.File(args.input, 'r') as fi:

        # fiPol0 = fiParent['pol0']
        # fiPol1 = fiParent['pol1']


        with h5py.File(output_file,'w') as fo:

            # Add attributes to group
            print("-| Copying attributes")
            for key, value in fi.attrs.items():
                fo.attrs[key] = value
                print("--| key: {}  | value: {}".format(key, value))

            print("-| Copying frequency and time arrays")
            fo.create_dataset('times', (len(fi['times']),), dtype="float64")[:] = fi['times']
            fo.create_dataset('freqs', (len(fi['freqs']),), dtype="float32")[:] = fi['freqs']


            # Creating output size
            output_shape = (fi['pol0'].shape[0], fi['pol0'].shape[0], fi['pol0'].shape[1], fi['pol0'].shape[2])
            print("-| Output ACM shape is: {}".format(output_shape))

            # Create a subdataset for each polarization
            print("-| Creating datasets full of zeros")
            foPol0_ACM = fo.create_dataset("pol0_ACM", output_shape, dtype=np.complex64)
            foPol1_ACM = fo.create_dataset("pol1_ACM", output_shape, dtype=np.complex64)

            ACM = np.zeros((2, 2, tlen, fft_size), dtype=phys_chan_1.dtype)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates an ACM from integrated spectra.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-i', '--input', type=str,
                        help='input h5 file of time series')
    args = parser.parse_args()
    main(args)