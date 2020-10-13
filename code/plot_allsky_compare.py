#!/usr/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
import pickle


def main(args):

    f1=h5py.File(args.file1, 'r')
    f2=h5py.File(args.file2, 'r')

    l1  = f1['l_est']
    m1  = f1['m_est']
    el1 = f1['elevation']
    az1 = f1['azimuth']
    h1  = f1['height']
    
    l2  = f2['l_est']
    m2  = f2['m_est']
    el2 = f2['elevation']
    az2 = f2['azimuth']
    h2  = f2['height']

    fig, ax = plt.subplots()

    for k in range(len(l1)): # doesn't matter which one
        print("Checking iteration {}".format(k))
        gridded_pkl = pathlib.Path(args.pkl_dir+"gridded_allsky_int_{}.pkl".format(k))
        if gridded_pkl.exists():
            print("- Imaging iteration {}".format(k))
            with open(gridded_pkl,'rb') as f:
                mydict = pickle.load(f)
                img = mydict['image']
                extent=mydict['extent']
            ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
            ax.plot(l1[k], m1[k], marker='o', color='r', label=args.label1)
            ax.plot(l2[k], m2[k], marker='X', color='k', label=args.label2)
            ax.set_ylabel(r'$m$')
            ax.set_xlabel(r'$l$')
            plt.legend(loc='lower right')
            plt.savefig('{}.png'.format(k))
            plt.cla()
        else:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Plot both models",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('--file1', '-f1', type=str,
            help='name of first HDF5 file')
    parser.add_argument('--file2', '-f2', type=str,
            help='name of second HDF5 file')   
    parser.add_argument('--label1', '-l1', type=str,
            help='label for first HDF5 file')  
    parser.add_argument('--label2', '-l2', type=str,
            help='label for second HDF5 file')
    parser.add_argument('--pkl-dir', '-d', type=str,
            help='name of directory storing pickled gridded images')   

    args = parser.parse_args()
    main(args)


