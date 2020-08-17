#!/usr/bin/python2

import argparse
import numpy as np
import h5py
from datetime import datetime
from scipy.optimize import least_squares
import plotly.graph_objects as go
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from generate_visibilities import compute_visibilities_gen, select_antennas
import known_transmitters
from visibility_models import point_residual_abs, point_residual_cplx, point_source_visibility_model

#tbn_filename = "../../data/058846_00123426_s0020.tbn"
#target_freq = 5351500
#transmitter_coords = get_transmitter_coords('SFe')
#l_guess = 0.0035 # from a manual fit to the first integration
#m_guess = 0.007

#tbn_filename = "../../data/058628_001748318.tbn"
#target_freq = 10e6
#transmitter_coords = get_transmitter_coords('WWV')
#l_guess = 0.008 # from a manual fit to the first integration
#m_guess = 0.031

# to be made into args:
fft_len = 16
use_pfb = False
use_pol = 0
integration_length = 1
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

def lm_to_ea(l, m):
    elev = np.arctan(m/l)
    
    azimuth = np.pi/2 - np.arccos(l/np.cos(elev))

    return elev, azimuth

def main(args):
    transmitter_coords = known_transmitters.parse_args(args)
    if transmitter_coords:
        bearing, _, distance = station.getPointingAndDistance(transmitter_coords + [0])
    else:
        print("Please specify a transmitter location")
        return

    tbnf = LWASVDataFile(args.tbn_filename)
    
    antennas = station.getAntennas()

    valid_ants, n_baselines = select_antennas(antennas, use_pol)

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
        ats['sample_rate'] = tbnf.getInfo('sampleRate')
        ats['start_time'] = str(datetime.utcfromtimestamp(tbnf.getInfo('tStart')))
        ats['valid_ants'] = [a.id for a in valid_ants]
        ats['n_baselines'] = n_baselines
        ats['center_freq'] = tbnf.getInfo('freq1')
        ats['res_function'] = residual_function.__name__
        ats['opt_method'] = opt_method
        # TODO: use cmd line parametersfor these
        ats['fft_len'] = fft_len
        ats['use_pfb'] = use_pfb
        ats['use_pol'] = use_pol
        ats['int_length'] = integration_length

        n_samples = tbnf.getInfo('nFrames') / tbnf.getInfo('nAntenna')
        samples_per_integration = int(integration_length * tbnf.getInfo('sampleRate') / 512)
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

    for bl, vis in compute_visibilities_gen(tbnf, valid_ants, args.tx_freq, integration_length=integration_length, fft_length=fft_len, use_pol=use_pol, use_pfb=use_pfb):
        
        # extract the baseline measurements from the baseline object pairs
        bl2d = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in bl])
        u = bl2d[:, 0]
        v = bl2d[:, 1]

        # we're only fitting the phase, so normalize the visibilities
        vis = vis/np.abs(vis)

        # start the optimization at the mean point of the 10 most recent fits
        l_init = l_est[-param_guess_av_length:].mean()
        m_init = m_est[-param_guess_av_length:].mean()

        print("Optimizing")
        opt_result = least_squares(
                residual_function,
                [l_init, m_init], 
                args=(u, v, vis),
                method=opt_method
                )

        print("Optimization result: {}".format(opt_result))
        print("Start point: {}".format((l_init, m_init)))
        l_out, m_out = opt_result['x']
        cost = opt_result['cost']

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

        elev, az = lm_to_ea(l_out, m_out)

        # use the flat mirror ionosphere model for now
        # TODO: tilted mirror model?
        height = (distance/2) * np.tan(elev)

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
            data = [
                go.Scatter3d(x=u, y=v, z=np.angle(vis), mode='markers', marker=dict(size=1, color='red')),
                go.Scatter3d(x=u, y=v, z=np.angle(point_source_visibility_model(u, v, l_out, m_out)), mode='markers', marker=dict(size=1, color='black'))
            ]

            fig = go.Figure(data=data)

            fig.update_layout(scene=dict(
                xaxis_title='u',
                yaxis_title='v',
                zaxis_title='phase'),
                title="Integration {}".format(k))

            fig.write_html("{}_scatter_int_{}.html".format(args.hdf5_file.split('/')[-1].split('.')[0], k))

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
    parser.add_argument('--scatter', type=int, nargs='*',
            help='export scatter plots for these integrations - warning: each scatter plot is about 6MB')
    parser.add_argument('--scatter_every', type=int,
            help='export a scatter plot every x integrations')
    #parser.add_argument('--scatter_bad_fits', action='store_true',
    #        help='export a scatter plot when the cost threshold is exceeded')
    parser.add_argument('--exclude', type=int, nargs='*',
            help="don't use these integrations in parameter guessing")
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
