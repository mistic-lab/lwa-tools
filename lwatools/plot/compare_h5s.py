#!/usr/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py



def main(args):

    f1=h5py.File(args.file1, 'r')
    int_time1 = f1.attrs['int_length']
    # need to index the HDF5 group to get a numpy array out of it
    l1  = f1['l_est'][:]
    m1  = f1['m_est'][:]
    el1 = f1['elevation'][:]
    az1 = f1['azimuth'][:]
    h1  = f1['height'][:]
    f1_domain = np.arange(0, len(l1) * int_time1, int_time1)
    fig, ax = plt.subplots(5,1)

    if args.file2 is not None:
        f2=h5py.File(args.file2, 'r')
        int_time2 = f2.attrs['int_length']
        l2  = f2['l_est'][:]
        m2  = f2['m_est'][:]
        el2 = f2['elevation'][:]
        az2 = f2['azimuth'][:]
        h2  = f2['height'][:]
        f2_domain = np.arange(0, len(l2) * int_time2, int_time2)

    ax[0].set_title(r'$l$')
    ax[1].set_title(r'$m$')
    ax[2].set_title('Elevation')
    ax[3].set_title('Azimuth')
    ax[4].set_title('Height')

    if args.diff:
        if not args.file2 or args.file3:
            raise RuntimeError("Need to supply file1 and file2 but not file3 for --diff")
        if int_time1 != int_time2:
            raise RuntimeError("Integration times must be equal to plot difference")

        ax[0].plot(f1_domain, l1 - l2, 'k', label=f"{args.label1} - {args.label2}")
        ax[1].plot(f1_domain, m1 - m2, 'k')
        ax[2].plot(f1_domain, el1 - el2, 'k')
        ax[3].plot(f1_domain, az1 - az2, 'k')
        ax[4].plot(f1_domain, h1 - h2, 'k')
    else: 
        ax[0].plot(f1_domain, l1, 'r', label=args.label1)
        ax[1].plot(f1_domain, m1, 'r')
        ax[2].plot(f1_domain, el1, 'r')
        ax[3].plot(f1_domain, az1, 'r')
        ax[4].plot(f1_domain, h1, 'r')

        if args.file2 is not None:
            if args.label2 is None:
                 raise RuntimeError("Need a label '-l2' for file2")
            ax[0].plot(f2_domain, l2, 'k', label=args.label2)
            ax[1].plot(f2_domain, m2, 'k')
            ax[2].plot(f2_domain, el2, 'k')
            ax[3].plot(f2_domain, az2, 'k')
            ax[4].plot(f2_domain, h2, 'k')

        if args.file3 is not None:
            if args.label3 is None:
                 raise RuntimeError("Need a label '-l3' for file3")
            f3=h5py.File(args.file3, 'r')
            int_time3 = f3.attrs['int_length']
            l3  = f3['l_est'][:]
            m3  = f3['m_est'][:]
            el3 = f3['elevation'][:]
            az3 = f3['azimuth'][:]
            h3  = f3['height'][:]
            f3_domain = np.arange(0, len(l3) * int_time3, int_time3)

            ax[0].plot(f3_domain, l3, 'b', label=args.label2)
            ax[1].plot(f3_domain, m3, 'b')
            ax[2].plot(f3_domain, el3, 'b')
            ax[3].plot(f3_domain, az3, 'b')
            ax[4].plot(f3_domain, h3, 'b')

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
    parser.add_argument('--diff', '-d', action='store_true',
            help='plot difference between file1 and file2')


    args = parser.parse_args()
    main(args)


