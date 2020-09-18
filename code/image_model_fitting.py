#!/usr/bin/python3

import argparse
import numpy as np
import h5py
from datetime import datetime
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile
from lsl.imaging.data import VisibilityDataSet, PolarizationDataSet
from lsl.imaging.analysis import find_point_sources
from lsl.imaging.utils import build_gridded_image, plot_gridded_image
from lsl.sim import vis as simVis

# from lsl.writer import fitsidi
# from lsl.correlator import fx as fxc
import matplotlib.pyplot as plt



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
    tbnf = LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True)
    
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
        ats['fft_len'] = args.fft_len
        ats['use_pfb'] = args.use_pfb
        ats['use_pol'] = args.use_pol
        ats['int_length'] = args.integration_length

        n_samples = tbnf.get_info('nframe') / tbnf.get_info('nantenna')
        samples_per_integration = int(args.integration_length * tbnf.get_info('sample_rate') / 512)
        n_integrations = n_samples / samples_per_integration
        h5f.create_dataset('l_est', (n_integrations,))
        h5f.create_dataset('m_est', (n_integrations,))
        h5f.create_dataset('elevation', (n_integrations,))
        h5f.create_dataset('azimuth', (n_integrations,))
        h5f.create_dataset('height', (n_integrations,))
#         h5f.create_dataset('skipped', (n_integrations,), dtype='bool')

    else:
        print("No output file specified.")
        return

    k = 0

    fig, ax = plt.subplots()
    for bl, freqs, vis in compute_visibilities_gen(tbnf, valid_ants, integration_length=args.integration_length, fft_length=args.fft_len, use_pol=args.use_pol, use_pfb=args.use_pfb):

        # we only want the bin nearest to our frequency
        target_bin = np.argmin([abs(args.tx_freq - f) for f in freqs])

        # Build a VisibilityDataSet with this data (lsl.imaging.data.VisibilityDataSet)
        # use_pol is 0 which fxc.pol_to_pols would return. pol_to_pols takes a string 'X' or 'XX' and outputs a 0.
        jd = tbnf.get_info('start_time').jd

        # Build antenna array
        antenna_array = simVis.build_sim_array(station, antennas, freqs/1e9, jd=jd)

        # build uvw
        uvw_m = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y, b[0].stand.z - b[1].stand.z]) for b in bl])

        uvw = np.empty((len(bl), 3, len(freqs)))
        for i, f in enumerate(freqs):
            wavelength = 3e8/f
            uvw[:,:,i] = uvw_m/wavelength

        dataSet = VisibilityDataSet(jd=jd, freq=freqs, baselines=bl, uvw=uvw, antennarray=antenna_array)
        if args.use_pol == 0:
            pol_string = 'XX'
            p=0 # this is related to the enumerate in lsl.imaging.utils.CorrelatedIDI().get_data_set() (for when there are multiple pols in a single dataset)
        else:
            raise RuntimeError("Only pol. XX supported right now.")
        polDataSet = PolarizationDataSet(pol_string, data=vis)
        dataSet.append(polDataSet)

        # Use lsl.imaging.utils.build_gridded_image (takes a VisibilityDataSet)
        #* This could become higher resolution by setting more parameters!
        gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin)

        # Plot/extract l/m do some modelling
        # I've largely borrow this from plot_gridded_image
        img = gridded_image.image()
        imgSize = img.shape[0]
        img = np.roll(img, imgSize//2, axis=0)
        img = np.roll(img, imgSize//2, axis=1)
        l, m = gridded_image.get_LM()
        l = np.linspace(l.min(), l.max(), img.shape[0])
        m = np.linspace(m.min(), m.max(), img.shape[1])
        if l.shape != m.shape:
            raise RuntimeError("gridded_image is not a square")

        row, col = np.where(img == img.max())
        if len(row) == 1 and len(col) == 1:
            row = row[0]
            col = col[0]
        else:
            raise RuntimeError("Nick messed up. There are two maxes. This method won't work.")

        # l = l[row]
        # m = m[col]
        l = l[-col]
        m = m[row]

        # Compute other values of interest
        elev, az = lm_to_ea(l, m)
        height = (distance/2) * np.tan(elev)

        h5f['l_est'][k] = l
        h5f['m_est'][k] = m
        h5f['elevation'][k] = elev
        h5f['azimuth'][k] = az
        h5f['height'][k] = height


        # l.dump("l_int_{}.npy".format(k))
        # m.dump("m_int_{}.npy".format(k))

        # (x, y, flux, sharpness, roundness) = find_point_sources(img)
        # print(f"x: {x} \n y: {y} \n flux: {flux} \n sharpness: {sharpness} \n roundness: {roundness}")


        save_all_sky = (args.all_sky and k in args.all_sky) #or (args.all_sky_every and k % args.all_sky_every == 0)# or (args.scatter_bad_fits and skip)
        if save_all_sky:
            plot_gridded_image(ax, gridded_image)
            plt.show()
            # np.save("allsky_int_{}.npy".format(k), img)

        k += 1


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
    parser.add_argument('--hdf5_file', '-f', type=str,
            help='name of output HDF5 file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--fft_len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use_pfb', type=bool, default=False,
            help='Whether to use PFB in correlator')
    parser.add_argument('--use_pol', type=int, default=0,
            help='Jeff what is this')
    parser.add_argument('--integration_length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--all-sky', type=int, nargs='*',
            help='export all-sky plots for these integrations')
    parser.add_argument('--all-sky-every', type=int,
            help='export an all-sky plot every x integrations')
    #parser.add_argument('--scatter_bad_fits', action='store_true',
    #        help='export a scatter plot when the cost threshold is exceeded')
#     parser.add_argument('--exclude', type=int, nargs='*',
#             help="don't use these integrations in parameter guessing")
#     parser.add_argument('--export_npy', action='store_true',
#             help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
