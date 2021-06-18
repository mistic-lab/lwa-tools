#!/usr/bin/python3
'''
run this script with -h for help
'''

import argparse
import numpy as np

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from lwatools.file_tools.outputs import build_output_file
from lwatools.utils.geometry import lm_to_ea
from lwatools.utils.array import select_antennas
from lwatools.utils import known_transmitters
from lwatools.visibilities.generate import compute_visibilities_gen
from lwatools.visibilities.baselines import uvw_from_antenna_pairs
from lwatools.visibilities.models import point_residual_abs, bind_gaussian_residual
from lwatools.visibilities.model_fitting import fit_model_to_vis
from lwatools.plot.vis import vis_phase_scatter_3d

opt_method = 'lm'

param_guess_av_length = 10

station = stations.lwasv

def main(args):

    print("Opening TBN file ({})".format(args.tbn_filename))
    with LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True) as tbnf:
    
        antennas = station.antennas

        valid_ants, n_baselines = select_antennas(antennas, args.use_pol)

        tx_coords = known_transmitters.parse_args(args)

        if args.visibility_model == 'point':
            residual_function = point_residual_abs
            residual_function_chain = None
        elif args.visibility_model == 'gaussian':
            residual_function = bind_gaussian_residual(1)
            residual_function_chain = None
        elif args.visibility_model == 'chained':
            residual_function = bind_gaussian_residual(0.5)
            residual_function_chain = point_residual_abs
        else:
            raise RuntimeError("Unknown visibility model option: {args.visibility_model}")

        if not args.hdf5_file:
            raise RuntimeError('Please provide an output filename')
        else:
            with build_output_file(args.hdf5_file, tbnf, valid_ants,
                    n_baselines, args.integration_length, tx_freq=args.tx_freq,
                    fft_len=args.fft_len, use_pfb=args.use_pfb,
                    use_pol=args.use_pol, opt_method=opt_method,
                    vis_model=args.visibility_model,
                    transmitter_coords=tx_coords) as h5f:

                # arrays for estimated parameters from each integration
                l_est = np.array([args.l_guess])
                m_est = np.array([args.m_guess])

                k = 0
                for bl, freqs, vis in compute_visibilities_gen(tbnf, valid_ants, integration_length=args.integration_length, fft_length=args.fft_len, use_pol=args.use_pol, use_pfb=args.use_pfb):

                    # start the optimization at the mean point of the 10 most recent fits
                    if args.visibility_model == 'point':
                        l_init = l_est[-param_guess_av_length:].mean()
                        m_init = m_est[-param_guess_av_length:].mean()
                    else:
                        l_init = 0
                        m_init = 0

                    target_bin = np.argmin([abs(args.tx_freq - f) for f in freqs])
                    
                    # TODO: is this correct? should it be the bin center?
                    uvw = uvw_from_antenna_pairs(bl, wavelength=3e8/args.tx_freq)

                    vis_tbin = vis[:, target_bin]

                    # do the model fitting to get parameter estimates
                    l_out, m_out, opt_result = fit_model_to_vis(uvw, vis_tbin, residual_function, 
                            l_init, m_init, export_npy=args.export_npy)

                    nfev = opt_result['nfev']

                    if residual_function_chain:
                        l_out, m_out, opt_result_chain = fit_model_to_vis(uvw, vis_tbin, residual_function_chain,
                                l_out, m_out, export_npy=args.export_npy)

                        nfev += opt_result_chain['nfev']

                    cost = opt_result['cost']

                    # see if we should skip including this in future starting parameter estimates
                    skip = False
                    if args.exclude and k in args.exclude:
                        print("Not including in parameter estimates by request")
                        skip = True

                    if not skip:
                        l_est = np.append(l_est, l_out)
                        m_est = np.append(m_est, m_out)
                        #costs = np.append(costs, cost)

                    # compute source sky location from parameter values
                    src_elev, src_az = lm_to_ea(l_out, m_out)

                    # write data to h5 file
                    h5f['l_start'][k] = l_init
                    h5f['m_start'][k] = m_init
                    h5f['l_est'][k] = l_out
                    h5f['m_est'][k] = m_out
                    h5f['elevation'][k] = src_elev
                    h5f['azimuth'][k] = src_az
                    h5f['cost'][k] = cost
                    h5f['skipped'][k] = skip
                    h5f['nfev'][k] = nfev

                    save_scatter = (args.scatter and k in args.scatter) or (args.scatter_every and k % args.scatter_every == 0)
                    if save_scatter:
                        print("Plotting model and data scatter")
                        vis_phase_scatter_3d(uvw[:,0], uvw[:,1], vis_tbin, show=False,
                                html_savename=f"scatter_{k}.html", l=l_out, m=m_out)

                    k += 1
                    print("\n\n")
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute visibilities and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--hdf5-file', '-f', type=str, default='output.h5',
            help='name of output HDF5 file')
    parser.add_argument('--l-guess', type=float, default=0.0,
            help='initial guess for l parameter')
    parser.add_argument('--m-guess', type=float, default=0.0,
            help='initial guess for m parameter')
    parser.add_argument('--fft-len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use-pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use-pol', type=int, default=0,
            help='0 for X which is the only supported polarization')
    parser.add_argument('--integration-length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--scatter', type=int, nargs='*',
            help='export scatter plots for these integrations - warning: each scatter plot is about 6MB')
    parser.add_argument('--scatter-every', type=int,
            help='export a scatter plot every x integrations')
    parser.add_argument('--exclude', type=int, nargs='*',
            help="don't use these integrations in parameter guessing")
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
    parser.add_argument('--visibility-model', default='gaussian',
            choices=('point', 'gaussian', 'chained'),
            help='select what kind of model is fit to the visibility data')
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
