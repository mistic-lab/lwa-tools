#!/usr/bin/python3

import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from lwatools.file_tools.outputs import build_output_file
from lwatools.imaging.utils import get_gimg_max, get_gimg_center_of_mass, grid_visibilities
from lwatools.visibilities.generate import compute_visibilities_gen
from lwatools.utils.array import select_antennas
from lwatools.utils import known_transmitters
from lwatools.utils.geometry import lm_to_ea


def main(args):
    station = stations.lwasv

    tx_coords = known_transmitters.parse_args(args)

    print("Opening TBN file ({})".format(args.tbn_filename))
    with LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True) as tbnf:
    
        antennas = station.antennas

        valid_ants, n_baselines = select_antennas(antennas, args.use_pol)

        if not args.hdf5_file:
            raise RuntimeError('Please provide an output filename')
        else:
            with build_output_file(h5_fname=args.hdf5_file, tbnf=tbnf, tx_freq=args.tx_freq, 
                    valid_ants=valid_ants, n_baeslines=n_baselines, fft_len=args.fft_len, use_pfb=args.use_pfb, use_pol=args.use_pol, 
                    integration_length=args.integration_length, transmitter_coords=tx_coords) as h5f:

                if args.point_finding_alg == 'all' or args.point_finding_alg == 'peak':
                    h5f.create_dataset_like('l_peak', h5f['l_est'])
                    h5f.create_dataset_like('m_peak', h5f['m_est'])
                    h5f.create_dataset_like('elevation_peak', h5f['elevation'])
                    h5f.create_dataset_like('azimuth_peak', h5f['azimuth'])
                if args.point_finding_alg == 'all' or args.point_finding_alg == 'CoM':
                    h5f.create_dataset_like('l_CoM', h5f['l_est'])
                    h5f.create_dataset_like('m_CoM', h5f['m_est'])
                    h5f.create_dataset_like('elevation_CoM', h5f['elevation'])
                    h5f.create_dataset_like('azimuth_CoM', h5f['azimuth'])
                else:
                    raise NotImplementedError(f"Unrecognized point finding algorithm: {args.point_finding_alg}")
                del h5f['l_est']
                del h5f['m_est']
                del h5f['elevation']
                del h5f['azimuth']



                k = 0

                save_all_sky = (args.all_sky and k in args.all_sky) or (args.all_sky_every and k % args.all_sky_every == 0)# or (args.scatter_bad_fits and skip)

                if save_all_sky:
                    fig, ax = plt.subplots()

                for bl, freqs, vis in compute_visibilities_gen(tbnf, valid_ants, integration_length=args.integration_length, fft_length=args.fft_len, use_pol=args.use_pol, use_pfb=args.use_pfb):

                    gridded_image = grid_visibilities(bl, freqs, vis, args.tx_freq, station)

                    save_all_sky = (args.all_sky and k in args.all_sky) or (args.all_sky_every and k % args.all_sky_every == 0)

                    if args.point_finding_alg == 'all' or 'peak':
                        result = get_gimg_max(gridded_image, return_img=save_all_sky)
                        l = result[0]
                        m = result[1]
                        src_elev, src_az = lm_to_ea(l, m)
                        h5f['l_peak'][k] = l
                        h5f['m_peak'][k] = m
                        h5f['elevation_peak'][k] = src_elev
                        h5f['azimuth_peak'][k] = src_az

                    if args.point_finding_alg == 'all' or args.point_finding_alg == 'CoM':
                        result = get_gimg_center_of_mass(gridded_image, return_img=save_all_sky)
                        l = result[0]
                        m = result[1]
                        src_elev, src_az = lm_to_ea(l, m)
                        h5f['l_CoM'][k] = l
                        h5f['m_CoM'][k] = m
                        h5f['elevation_CoM'][k] = src_elev
                        h5f['azimuth_CoM'][k] = src_az

                    if save_all_sky:
                        img = result[2]
                        extent = result[3]
                        ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
                        plt.savefig('allsky_int_{}.png'.format(k))

                    k += 1
                    print("\n\n")
                    if args.stop_after >= 0 and k >= args.stop_after:
                        break
                




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute all-sky images and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--hdf5-file', '-f', type=str,
            help='name of output HDF5 file')
    parser.add_argument('--fft-len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use-pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use-pol', type=int, default=0,
            help='0 means X and is the only currently supported option')
    parser.add_argument('--integration-length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--all-sky', type=int, nargs='*',
            help='export all-sky plots for these integrations')
    parser.add_argument('--all-sky-every', type=int,
            help='export an all-sky plot every x integrations')
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
    parser.add_argument('--stop-after', type=int, default=-1,
            help='stop running after this many integrations')
    parser.add_argument('--point-finding-alg', nargs='?', default='all', choices=('peak', 'CoM', 'all'),
            help='select which algorithm is used to locate the point source in an image - options are the image peak or centre of mass')
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
