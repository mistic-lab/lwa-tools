#!/usr/bin/python3

import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from lwatools.file_tools.outputs import build_output_file
from lwatools.imaging.imaging_utils import lm_to_ea, get_gimg_max, get_gimg_center_of_mass, grid_visibilities
from lwatools.ionospheric_models.fixed_dist_mirrors import flatmirror_height, tiltedmirror_height
from lwatools.visibilities.generate import compute_visibilities_gen, select_antennas
from lwatools.utils import known_transmitters


def main(args):
    station = stations.lwasv

    tx_coords = known_transmitters.parse_args(args)
    if not tx_coords:
        print("Please specify a transmitter location")
        return

    rx_coords = [station.lat * 180/np.pi, station.lon * 180/np.pi]

    print(args)

    print("Opening TBN file ({})".format(args.tbn_filename))
    tbnf = LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True)
    
    antennas = station.antennas

    # valid_ants, n_baselines = select_antennas(antennas, args.use_pol)
    valid_ants, n_baselines = select_antennas(antennas, args.use_pol, exclude=[256]) # to exclude outrigger

    if args.hdf5_file:
        h5f = build_output_file(args.hdf5_file, tbnf, tx_coords, args.tx_freq, 
                valid_ants, n_baselines, args.fft_len, args.use_pfb, args.use_pol, 
                args.integration_length, "imaging", "")

    k = 0

    save_all_sky = (args.all_sky and k in args.all_sky) or (args.all_sky_every and k % args.all_sky_every == 0)# or (args.scatter_bad_fits and skip)

    if save_all_sky:
        fig, ax = plt.subplots()

    if args.point_finding_alg == 'peak':
        get_gimg = get_gimg_max
    elif args.point_finding_alg == 'CoM':
        get_gimg = get_gimg_center_of_mass
    else:
        raise NotImplementedError(f"unrecognized point finding algorithm: {args.point_finding_alg}")

    for bl, freqs, vis in compute_visibilities_gen(tbnf, valid_ants, integration_length=args.integration_length, fft_length=args.fft_len, use_pol=args.use_pol, use_pfb=args.use_pfb):

        # Normalize amplitudes since we want it based on phase
        # vis/=np.abs(vis)


        jd = tbnf.get_info('start_time').jd

        gridded_image = grid_visibilities(bl, freqs, vis, args.tx_freq, jd, station)


        save_all_sky = (args.all_sky and k in args.all_sky) or (args.all_sky_every and k % args.all_sky_every == 0)
        save_pkl_gridded = (args.pkl_gridded and k in args.pkl_gridded) or (args.pkl_gridded_every and k % args.pkl_gridded_every == 0)
        if save_all_sky==True or save_pkl_gridded==True:
            l, m, img, extent = get_gimg(gridded_image, return_img=True)
        else:
            l,m = get_gimg(gridded_image)

        # Compute other values of interest
        src_elev, src_az = lm_to_ea(l, m)

        if args.reflection_model == 'flat_fixed_dist':
            height = flatmirror_height(tx_coords, rx_coords, src_elev)
        elif args.reflection_model == 'tilted_fixed_dist':
            height = tiltedmirror_height(tx_coords, rx_coords, src_elev, src_az)
        else:
            raise NotImplementedError(f"unrecognized reflection model: {args.reflection_model}")

        if args.hdf5_file:
            h5f['l_est'][k] = l
            h5f['m_est'][k] = m
            h5f['elevation'][k] = src_elev
            h5f['azimuth'][k] = src_az
            h5f['height'][k] = height

        if args.export_npy:
            print("Exporting u, v, w, and visibility")
            np.save('uvw{}.npy'.format(k), uvw)
            np.save('vis{}.npy'.format(k), vis)
            print("Exporting gridded u, v, and visibility")
            u,v = gridded_image.get_uv()
            np.save('gridded-u{}.npy'.format(k), u)
            np.save('gridded-v{}.npy'.format(k), v)
            np.save('gridded-vis{}.npy'.format(k), gridded_image.uv)

        if save_all_sky:
            ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
            # plot_gridded_image(ax, gridded_image)
            plt.savefig('allsky_int_{}.png'.format(k))

        if save_pkl_gridded:
            quickDict={'image':img, 'extent':extent}
            with open('gridded_allsky_int_{}.pkl'.format(k),'wb') as f:
                pickle.dump(quickDict, f, protocol=pickle.HIGHEST_PROTOCOL)

        k += 1
        if k>=args.stop_after:
            break

    if args.hdf5_file:
        h5f.close()
    tbnf.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute all-sky images and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('--hdf5-file', '-f', type=str,
            help='name of output HDF5 file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--fft-len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use-pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use-pol', type=int, default=0,
            help='Jeff what is this')
    parser.add_argument('--integration-length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--all-sky', type=int, nargs='*',
            help='export all-sky plots for these integrations')
    parser.add_argument('--all-sky-every', type=int,
            help='export an all-sky plot every x integrations')
    parser.add_argument('--pkl-gridded', type=int, nargs='*',
            help='export gridded all sky data for these integrations')
    parser.add_argument('--pkl-gridded-every', type=int,
            help='export gridded all sky data every x integrations')
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
    parser.add_argument('--stop-after', type=int, default=999999999,
            help='stop running after this many integrations')
    parser.add_argument('--point-finding-alg', nargs='?', default='peak', choices=('peak', 'CoM'),
            help='select which algorithm is used to locate the point source in an image - options are the image peak or centre of mass')
    parser.add_argument('--reflection-model', nargs='?', default='flat_fixed_dist', 
            choices=('flat_fixed_dist', 'tilted_fixed_dist'),
            help='select which ionospheric model is used to convert DoA into virtual height - flat and tilted halfway-point mirror models are available')
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
