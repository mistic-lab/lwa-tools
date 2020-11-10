#!/usr/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py



def main(args):

    f1=h5py.File(args.file1, 'r')

    l1  = f1['l_est']
    m1  = f1['m_est']
    el1 = f1['elevation']
    az1 = f1['azimuth']
    h1  = f1['height']
    fig, ax = plt.subplots(5,1)

    ax[0].set_title(r'$l$')
    ax[0].plot(l1, 'r', label=args.label1)

    ax[1].set_title(r'$m$')
    ax[1].plot(m1, 'r')

    ax[2].set_title('Elevation')
    ax[2].plot(el1, 'r')

    ax[3].set_title('Azimuth')
    ax[3].plot(az1, 'r')

    ax[4].set_title('Height')
    ax[4].plot(h1, 'r')

    if args.file2 is not None:
        if args.label2 is None:
             raise RuntimeError("Need a label '-l2' for file2".)
        f2=h5py.File(args.file2, 'r')
        l2  = f2['l_est']
        m2  = f2['m_est']
        el2 = f2['elevation']
        az2 = f2['azimuth']
        h2  = f2['height']

        ax[0].plot(l2, 'k', label=args.label2)
        ax[1].plot(m2, 'k')
        ax[2].plot(el2, 'k')
        ax[3].plot(az2, 'k')
        ax[4].plot(h2, 'k')

    if args.file3 is not None:
        if args.label3 is None:
             raise RuntimeError("Need a label '-l3' for file3".)
        f3=h5py.File(args.file3, 'r')
        l3  = f3['l_est']
        m3  = f3['m_est']
        el3 = f3['elevation']
        az3 = f3['azimuth']
        h3  = f3['height']

        ax[0].plot(l3, 'b', label=args.label2)
        ax[1].plot(m3, 'b')
        ax[2].plot(el3, 'b')
        ax[3].plot(az3, 'b')
        ax[4].plot(h3, 'b')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Plot output from multiple hdf5 files",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('--file1', '-f1', type=str,
            help='name of first HDF5 file')
    parser.add_argument('--file2', '-f2', type=str, default=None,
            help='name of second HDF5 file')   
    parser.add_argument('--file3', '-f3', type=str, default=None,
            help='name of possible third HDF5 file')   
    parser.add_argument('--label1', '-l1', type=str,
            help='label for first HDF5 file')  
    parser.add_argument('--label2', '-l2', type=str, default=None,
            help='label for second HDF5 file')
    parser.add_argument('--label3', '-l3', type=str, default=None,
            help='label for possible third HDF5 file')


    args = parser.parse_args()
    main(args)


