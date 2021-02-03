#!/usr/bin/python3

import argparse
import numpy as np
import h5py
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile
from lsl.imaging.data import VisibilityDataSet, PolarizationDataSet
from lsl.imaging.analysis import find_point_sources
from lsl.imaging.utils import build_gridded_image, plot_gridded_image
from lsl.sim import vis as simVis
# from lsl.writer import fitsidi
# from lsl.correlator import fx as fxc

from lwatools.file_tools.outputs import build_output_file
from lwatools.vis_modeling.visibility_models import point_source_visibility_model_uv
from lwatools.vis_modeling.baselines import uvw_from_antenna_pairs
from lwatools.imaging.imaging_utils import lm_to_ea, flatmirror_height, get_gimg_max
from lwatools.vis_modeling.generate_visibilities import compute_visibilities_gen, select_antennas
from lwatools.utils import known_transmitters

def grid_visibilities(bl, freqs, vis, tx_freq, jd, valid_ants, station, size=80, res=0.5, wres=0.10, use_pol=0):
    '''
    Resamples the baseline-sampled visibilities on to a regular grid. 

    arguments:
    bl = pairs of antenna objects representing baselines (list)
    freqs = frequency channels for which we have correlations (list)
    vis = visibility samples corresponding to the baselines (numpy array)
    target_bin = the bin we want to use (number)
    jd = the date - shouldn't actually be important, but VisibilityDataSet needs it (number)
    valid_ants = which antennas we actually want to use (list)
    station = lsl station object - usually stations.lwasv
    according to LSL docstring:
        size = number of wavelengths which the UV matrix spans (this 
        determines the image resolution).
        res = resolution of the UV matrix (determines image field of view).
        wres: the gridding resolution of sqrt(w) when projecting to w=0.

    use_pol = which polarization to use (only 0 is supported right now)
    returns:
    gridded_image
    '''
    # In order to do the gridding, we need to build a VisibilityDataSet using
    # lsl.imaging.data.VisibilityDataSet. We have to build a bunch of stuff to
    # pass to its constructor.

    # we only want the bin nearest to our frequency
    target_bin = np.argmin([abs(tx_freq - f) for f in freqs])

    # Build antenna array
    antenna_array = simVis.build_sim_array(station, valid_ants, freqs/1e9, jd=jd, force_flat=True)

    uvw = np.empty((len(bl), 3, len(freqs)))

    for i, f in enumerate(freqs):
        # wavelength = 3e8/f # TODO this should be fixed. What is currently happening is not true. Well it is, but only if you're looking for a specific transmitter frequency. Which I guess we are. I just mean it's not generalized.
        wavelength = 3e8/tx_freq
        uvw[:,:,i] = uvw_from_antenna_pairs(bl, wavelength=wavelength)


    dataSet = VisibilityDataSet(jd=jd, freq=freqs, baselines=bl, uvw=uvw, antennarray=antenna_array)
    if use_pol == 0:
        pol_string = 'XX'
        p=0 # this is related to the enumerate in lsl.imaging.utils.CorrelatedIDI().get_data_set() (for when there are multiple pols in a single dataset)
    else:
        raise RuntimeError("Only pol. XX supported right now.")
    polDataSet = PolarizationDataSet(pol_string, data=vis)
    dataSet.append(polDataSet)


    # Use lsl.imaging.utils.build_gridded_image (takes a VisibilityDataSet)
    gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin, size=size, res=res, wres=wres)

    # gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin, size=100, res=0.3) #what I think it had ought to be if size=N and res=du
    # gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin, size=3, res=0.01) #what I think it had ought to be if the docstring is true
    # gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin, size=10, res=0.05) #what I think it had ought to be if the docstring is true
    # gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin, size=20, res=0.5) #from sim

    return gridded_image

def main(args):
    station = stations.lwasv

    transmitter_coords = known_transmitters.parse_args(args)
    if transmitter_coords:
        bearing, _, distance = station.get_pointing_and_distance(transmitter_coords + [0])
    else:
        print("Please specify a transmitter location")
        return

    print("Opening TBN file ({})".format(args.tbn_filename))
    tbnf = LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True)
    
    antennas = station.antennas

    # valid_ants, n_baselines = select_antennas(antennas, args.use_pol)
    valid_ants, n_baselines = select_antennas(antennas, args.use_pol, exclude=[256]) # to exclude outrigger

    if args.hdf5_file:
        h5f = build_output_file(args.hdf5_file, tbnf, args.transmitter, args.tx_freq, 
                valid_ants, n_baselines, args.fft_len, args.use_pfb, args.use_pol, 
                args.integration_length, "imaging", "")

    k = 0

    save_all_sky = (args.all_sky and k in args.all_sky) or (args.all_sky_every and k % args.all_sky_every == 0)# or (args.scatter_bad_fits and skip)

    if save_all_sky:
        fig, ax = plt.subplots()

    for bl, freqs, vis in compute_visibilities_gen(tbnf, valid_ants, integration_length=args.integration_length, fft_length=args.fft_len, use_pol=args.use_pol, use_pfb=args.use_pfb):

        # Normalize amplitudes since we want it based on phase
        vis/=np.abs(vis)


        jd = tbnf.get_info('start_time').jd

        gridded_image = grid_visibilities(bl, freqs, vis, args.tx_freq, jd, valid_ants, station)


        save_all_sky = (args.all_sky and k in args.all_sky) or (args.all_sky_every and k % args.all_sky_every == 0)
        save_pkl_gridded = (args.pkl_gridded and k in args.pkl_gridded) or (args.pkl_gridded_every and k % args.pkl_gridded_every == 0)
        if save_all_sky==True or save_pkl_gridded==True:
            l, m, img, extent = get_gimg_max(gridded_image, return_img=True)
        else:
            l,m = get_gimg_max(gridded_image)

        # Compute other values of interest
        elev, az = lm_to_ea(l, m)
        height = flatmirror_height(elev, distance)

        if args.hdf5_file:
            h5f['l_est'][k] = l
            h5f['m_est'][k] = m
            h5f['elevation'][k] = elev
            h5f['azimuth'][k] = az
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
    parser.add_argument('--hdf5_file', '-f', type=str,
            help='name of output HDF5 file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--fft_len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use_pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use_pol', type=int, default=0,
            help='Jeff what is this')
    parser.add_argument('--integration_length', type=float, default=1,
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
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
