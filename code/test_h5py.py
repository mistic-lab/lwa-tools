##########
# Author: Nicholas Bruce
# Date: July 2019
#
# Checks whether an hdf5 file contains the same data (but sorted) as a raw tbn file.
#
##########


import parseTBN
import numpy as np
import h5py
import argparse


def main(args):

    h5pyFileName = args.hdf5
    tbnFileName = args.tbn
    decimation = args.decimation

    with h5py.File(h5pyFileName, 'r') as f:
        parent = f[f.keys()[0]] # Should only ever be one parent
        pol0 = parent['pol0']
        pol1 = parent['pol1']

        num_ants = pol0.shape[0]/decimation
        arr_length = pol0.shape[1]

        pol0_results = np.zeros(num_ants, dtype=bool)
        pol1_results = np.zeros(num_ants, dtype=bool)

        for setDict in [{'pol': 0, 'dset': pol0, 'results': pol0_results, 'title': "POL0"}, {'pol': 1, 'dset': pol1, 'results': pol1_results, 'title': "POL1"}]:
            pol = setDict['pol']
            dset = setDict['dset']
            title = setDict['title']
            results = setDict['results']

            for i in range(num_ants):

                tbnArray = parseTBN.extract_single_ant(tbnFileName, i+1, pol)
                tbnArray = tbnArray[:arr_length]

                results[i] = np.array_equal(tbnArray, dset[i,:])
                print("{}: {}".format(i, results[i]))

            unique, counts = np.unique(results, return_counts=True)
            dictResults = dict(zip(unique, counts))
        
            print(title)
            try:
                print("Number passed: {}".format(dictResults[True]))
            except KeyError:
                print("Number passed: 0")

            try:
                print("Number failed: {}".format(dictResults[False]))
            except KeyError:
                print("Number failed: 0")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='converts TBN file to hdf5 file', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('-h', 'hdf5', type=str, 
        help='hdf5 file to compare')
        parser.add_argument('-t', 'tbn', type=str, 
        help='tbn file to compare')
        parser.add_argument('-d', 'decimation', type=int, default=32,
        help='how many antennas to skip between checks')
args = parser.parse_args()
    main(args)
