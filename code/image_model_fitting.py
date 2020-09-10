#!/usr/bin/python3

import argparse
import numpy as np
import h5py
from datetime import datetime
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from generate_visibilities import compute_visibilities_gen, select_antennas
import known_transmitters

station=stations.lwasv

# TODO: This is in both vis and image fitting scripts. put it somewhere general.
def lm_to_ea(l, m):
    azimuth = np.pi/2 - np.arctan(m/l)
    
    elev = np.arccos(np.sqrt(l**2 + m**2))

    return elev, azimuth

def main(args):
    transmitter_coords = known_transmitters.parse_args(args)
    if transmitter_coords:
        bearing, _, distance = station.get_pointing_and_distance(transmitter_coords + [0])
    else:
        print("Please specify a transmitter location")
        return

    print("Opening TBN file ({})".format(args.tbn_filename))
    tbnf = LWASVDataFile(args.tbn_filename, ignore_time_tag_errors=True)
    
    antennas = station.antennas

    valid_ants, n_baselines = select_antennas(antennas, args.use_pol)

    if args.hdf5_file:
        print("Writing output to {}".format(args.hdf5_file))
        h5f = h5py.File(args.hdf5_file, 'w')

        # write metadata to attributes
        ats = h5f.attrs
        ats['tbn_filename'] = args.tbn_filename
        ats['transmitter'] = args.transmitter
        ats['tx_bearing'] = bearing
        ats['tx_distance'] = distance
        ats['tx_freq'] = args.tx_freq
        ats['sample_rate'] = tbnf.get_info('sample_rate')
        ats['start_time'] = str(tbnf.get_info('start_time').utc_datetime)
        ats['valid_ants'] = [a.id for a in valid_ants]
        ats['n_baselines'] = n_baselines
        ats['center_freq'] = tbnf.get_info('freq1')
        # ats['res_function'] = residual_function.__name__
        # ats['opt_method'] = opt_method
        ats['fft_len'] = args.fft_len
        ats['use_pfb'] = args.use_pfb
        ats['use_pol'] = args.use_pol
        ats['int_length'] = args.integration_length

        n_samples = tbnf.getInfo('nframe') / tbnf.get_info('nantenna')
        samples_per_integration = int(args.integration_length * tbnf.getInfo('sample_rate') / 512)
        n_integrations = n_samples / samples_per_integration
        h5f.create_dataset('l_start', (n_integrations,))
        h5f.create_dataset('m_start', (n_integrations,))
        h5f.create_dataset('l_est', (n_integrations,))
        h5f.create_dataset('m_est', (n_integrations,))
        h5f.create_dataset('elevation', (n_integrations,))
        h5f.create_dataset('azimuth', (n_integrations,))
        h5f.create_dataset('height', (n_integrations,))
        h5f.create_dataset('cost', (n_integrations,))
        h5f.create_dataset('skipped', (n_integrations,), dtype='bool')

    else:
        print("No output file specified.")
        return

    # arrays for estimated parameters from each integration
    l_est = np.array([args.l_guess])
    m_est = np.array([args.m_guess])
    #costs = np.array([])
    elev_est = np.array([])
    az_est = np.array([])
    height_est = np.array([])

    k = 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute all-sky images and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('--hdf5_file', '-f', type=str,
            help='name of output HDF5 file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('fft_len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('use_pfb', type=bool, default=False,
            help='Whether to use PFB in correlator')
    parser.add_argument('use_pol', type=int, default=0,
            help='Jeff what is this')
    parser.add_argument('integration_length', type=float, default=1,
            help='Integration length in seconds')

    parser.add_argument('l_guess', type=float,
            help='initial guess for l parameter')
    parser.add_argument('m_guess', type=float,
            help='initial guess for m parameter')
    parser.add_argument('--scatter', type=int, nargs='*',
            help='export scatter plots for these integrations - warning: each scatter plot is about 6MB')
    parser.add_argument('--scatter_every', type=int,
            help='export a scatter plot every x integrations')
    #parser.add_argument('--scatter_bad_fits', action='store_true',
    #        help='export a scatter plot when the cost threshold is exceeded')
    parser.add_argument('--exclude', type=int, nargs='*',
            help="don't use these integrations in parameter guessing")
    parser.add_argument('--export_npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
