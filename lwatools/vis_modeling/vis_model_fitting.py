#!/usr/bin/python3

import argparse
import numpy as np
import h5py
from datetime import datetime
from scipy.optimize import least_squares
import plotly.graph_objects as go
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from lwatools.imaging.imaging_utils import lm_to_ea, flatmirror_height
from lwatools.vis_modeling.generate_visibilities import compute_visibilities_gen, select_antennas
from lwatools.utils import known_transmitters
from lwatools.vis_modeling.visibility_models import point_residual_abs, point_residual_cplx, point_source_visibility_model_uvw
from lwatools.plot.vis import vis_phase_scatter_3d

residual_function = point_residual_abs
opt_method = 'lm'

param_guess_av_length = 10
#cost_threshold_av_length = 10
#cost_threshold_sigma = 3

station = stations.lwasv

def ls_cost(params, u, v, vis, resid=point_residual_abs):
    '''
    Computes the least-squares cost function at the given parameter values.
    least_squares actually takes care of this step, but this is here for
    visualization and debugging purposes.
    '''
    r = resid(params, u, v, vis)
    return np.dot(r,r)

def setup_results_h5(h5_fname, tbnf, transmitter, tx_freq, valid_ants, n_baselines, fft_len, use_pfb, use_pol, integration_length, opt_method, res_function_name):
    '''
    Opens the hdf5 file that will be used to store results, initializes
    datasets, and writes metadata.
    '''

    print("Writing output to {}".format(h5_fname))
    h5f = h5py.File(h5_fname, 'w')

    transmitter_coords = known_transmitters.get_transmitter_coords(transmitter[0])

    # write metadata to attributes
    ats = h5f.attrs
    ats['tbn_filename'] = tbnf.filename
    ats['transmitter'] = transmitter
    ats['tx_bearing'], _, ats['tx_distance'] = station.get_pointing_and_distance(transmitter_coords + [0])
    ats['tx_freq'] = tx_freq
    ats['sample_rate'] = tbnf.get_info('sample_rate')
    ats['start_time'] = str(tbnf.get_info('start_time').utc_datetime)
    ats['valid_ants'] = [a.id for a in valid_ants]
    ats['n_baselines'] = n_baselines
    ats['center_freq'] = tbnf.get_info('freq1')
    ats['fft_len'] = fft_len
    ats['use_pfb'] = use_pfb
    ats['use_pol'] = use_pol
    ats['int_length'] = integration_length
    ats['opt_method'] = opt_method
    ats['res_function'] = res_function_name 

    n_samples = tbnf.get_info('nframe') / tbnf.get_info('nantenna')
    samples_per_integration = int(integration_length * tbnf.get_info('sample_rate') / 512)
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

    return h5f

def fit_model_to_vis(bl, freqs, vis, target_bin, residual_function, l_init, m_init,
        opt_method='lm', export_npy=False, param_guess_av_length=10):
    '''
    Fits a point source (or equivalently a gaussian) model to the visibilities in vis.

    It's monochromatic (single-frequency) for now.

    l_est, m_est should be arrays of previous l,m values, the mean of which is
    used as an optimization starting point.

    returns l, m, cost
    ( the optimized l,m parameter values and the cost function after optimization )
    '''

    # monochromatic for now
    # TODO: make it not monochromatic
    vis = vis[:, target_bin]
    freqs = freqs[target_bin]

    # extract the baseline measurements from the baseline object pairs
    bl2d = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y, b[0].stand.z-b[1].stand.z]) for b in bl])
    u = bl2d[:, 0]
    v = bl2d[:, 1]
    w = bl2d[:, 2]

    # convert the baselines to wavelenths -- great job jeff
    wavelength = 3e8/args.tx_freq

    u = u/wavelength
    v = v/wavelength
    w = w/wavelength

    # we're only fitting the phase, so normalize the visibilities
    vis = vis/np.abs(vis)

    if export_npy:
        print("Exporting u, v, w, and visibility")
        np.save('u{}.npy'.format(k), u)
        np.save('v{}.npy'.format(k), v)
        np.save('w{}.npy'.format(k), w)
        np.save('vis{}.npy'.format(k), vis)


    print("Optimizing")
    opt_result = least_squares(
            residual_function,
            [l_init, m_init], 
            args=(u, v, w, vis),
            method=opt_method
            )

    print("Optimization result: {}".format(opt_result))
    print("Start point: {}".format((l_init, m_init)))
    l_out, m_out = opt_result['x']
    cost = opt_result['cost']

    return l_out, m_out, cost


def main(args):

    print("Opening TBN file ({})".format(args.tbn_filename))
    tbnf = LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True)
    
    antennas = station.antennas

    valid_ants, n_baselines = select_antennas(antennas, args.use_pol, exclude=[256]) # to exclude outrigger

    transmitter_coords = known_transmitters.parse_args(args)
    _, _, distance = station.get_pointing_and_distance(transmitter_coords + [0])

    if args.hdf5_file:
        h5f = setup_results_h5(args.hdf5_file, tbnf, args.transmitter, args.tx_freq, 
                valid_ants, n_baselines, args.fft_len, args.use_pfb, args.use_pol, 
                args.integration_length, opt_method, residual_function.__name__)
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
    for bl, freqs, vis in compute_visibilities_gen(tbnf, valid_ants, integration_length=args.integration_length, fft_length=args.fft_len, use_pol=args.use_pol, use_pfb=args.use_pfb):

        # we only want the bin nearest to our frequency
        target_bin = np.argmin([abs(args.tx_freq - f) for f in freqs])

        # start the optimization at the mean point of the 10 most recent fits
        l_init = l_est[-param_guess_av_length:].mean()
        m_init = m_est[-param_guess_av_length:].mean()
        

        # do the model fitting to get parameter estimates
        l_out, m_out, cost = fit_model_to_vis(bl, freqs, vis, target_bin,
                residual_function, l_init, m_init, export_npy=args.export_npy)

        # see if we should skip including this in future starting parameter estimates
        skip = False
        if args.exclude and k in args.exclude:
            print("Not including in parameter estimates by request")
            skip = True
        #elif len(costs) > 10:
        #    recent_costs = costs[-cost_threshold_av_length:]
        #    if cost > (recent_costs.mean() + cost_threshold_sigma * recent_costs.std()):
        #        print("Not including in parameter estimates due to cost")
        #        skip = True

        if not skip:
            l_est = np.append(l_est, l_out)
            m_est = np.append(m_est, m_out)
            #costs = np.append(costs, cost)

        # compute source sky location from parameter values
        elev, az = lm_to_ea(l_out, m_out)

        # TODO: tilted mirror model?
        height = flatmirror_height(elev, distance)

        # write data to h5 file
        h5f['l_start'][k] = l_init
        h5f['m_start'][k] = m_init
        h5f['l_est'][k] = l_out
        h5f['m_est'][k] = m_out
        h5f['elevation'][k] = elev
        h5f['azimuth'][k] = az
        h5f['cost'][k] = cost
        h5f['height'][k] = height
        h5f['skipped'][k] = skip

        save_scatter = (args.scatter and k in args.scatter) or (args.scatter_every and k % args.scatter_every == 0)# or (args.scatter_bad_fits and skip)
        if save_scatter:
            print("Plotting model and data scatter")
            vis_phase_scatter_3d(u, v, vis, l_out, m_out)

        k += 1
        print("\n\n")
  
    h5f.close()
    tbnf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute visibilities and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('--hdf5_file', '-f', type=str,
            help='name of output HDF5 file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('l_guess', type=float,
            help='initial guess for l parameter')
    parser.add_argument('m_guess', type=float,
            help='initial guess for m parameter')
    parser.add_argument('--fft_len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use_pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use_pol', type=int, default=0,
            help='Jeff what is this')
    parser.add_argument('--integration_length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--scatter', type=int, nargs='*',
            help='export scatter plots for these integrations - warning: each scatter plot is about 6MB')
    parser.add_argument('--scatter_every', type=int,
            help='export a scatter plot every x integrations')
    parser.add_argument('--exclude', type=int, nargs='*',
            help="don't use these integrations in parameter guessing")
    parser.add_argument('--export_npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
