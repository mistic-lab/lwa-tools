import h5py
import argparse
from os.path import isfile

from lsl.common import stations

from lwatools.ionospheric_models.fixed_dist_mirrors import flatmirror_height, tiltedmirror_height

def main(args):
    
    # make sure the file exists
    if not isfile(args.hdf5_filename):
        raise RuntimeError(f"File {fname} does not exist")

    # open the file
    h5f = h5py.File(args.hdf5_filename, 'a')

    # get the info we need from the file attributes
    tx_coords = h5f.attrs['transmitter_coords']
    # TODO: store rx coords in h5 files so we don't depend on lsl here
    rx_coords = [stations.lwasv.lat * 180/np.pi, stations.lwasv.lon * 180/np.pi]

    # the new dataset will be as long as the elevation data in the file
    d_len = len(h5f['elevation'])

    if 'flat_halfway' in args.selected_models:
        dataset_name = 'height_flat_halfway'
        print(f"Processing fixed-distance flat mirror model (new dataset: {dataset_name})")

        if dataset_name in h5f.keys():
            msg = f"{dataset_name} dataset already exists in {h5f.filename}"
            if args.overwrite:
                print(msg + ", overwriting...")
            else:
                raise RuntimeError(msg)
        else:
            h5f.create_dataset(dataset_name, d_len)

        h5f[dataset_name][:] = flatmirror_height(tx_coords, rx_coords, h5f['elevation'][:])

    if 'tilted_halfway' in args.selected_models:
        dataset_name = 'height_tilted_halfway'
        print(f"Processing fixed-distance tilted mirror model (new dataset: {dataset_name})")

        if dataset_name in h5f.keys():
            msg = f"{dataset_name} dataset already exists in {h5f.filename}"
            if args.overwrite:
                print(msg + ", overwriting...")
            else:
                raise RuntimeError(msg)
        else:
            h5f.create_dataset(dataset_name, d_len)

        h5f[dataset_name][:] = tiltedmirror_height(tx_coords, rx_coords, h5f['elevation'][:], h5f['azimuth'][:])

    print("Done")
    h5f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Adds height datasets using the specified model(s) to HDF5 files containing elevation and azimuth data',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('--overwrite', action='store_true',
            help='If the datasets to be generated already exist in the file, then overwrite them')
    parser.add_argument('hdf5_filename', type=str,
            help='Name of HDF5 file to process')
    parser.add_argument('selected_models', nargs='*', type=str,
            choices=('flat_halfway', 'tilted_halfway'),
            help='Which model(s) to apply to the data. One new dataset will be created for each.')

    args = parser.parse_args()
    main(args)
