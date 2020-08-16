#!/usr/bin/python2

import argparse
import numpy as np
from scipy.optimize import least_squares
import plotly.graph_objects as go
from lsl.common import stations

from generate_visibilities import compute_visibilities_gen
import known_transmitters

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

station = stations.lwasv

def point_source_visibility_model(u, v, l, m):
    '''
    Computes visibility at u,v as if from a perfect point source located at (l,m)
    '''
    return np.exp(2j * np.pi * (l*u + v*m))

def residual_cplx(params, u, v, vis):
    '''
    Calculates the residuals of the model fit with the differences of the real
    and imaginary parts each giving one residual.
    '''
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model(u, v, l, m)

    return np.concatenate([mc.real, mc.imag]) - np.concatenate([vis.real, vis.imag])


def residual_abs(params, u, v, vis):
    '''
    Calculates the residual of the model fit as the magnitude of the difference
    between the model and the actual visibilities.
    '''
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model(u, v, l, m)

    return np.abs(mc - vis)


def ls_cost(params, u, v, vis, resid=residual_abs):
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

    if args.csv:
        print("Writing CSV output to {}".format(args.csv))
        csvf = open(args.csv, 'w')
        csvf.write("integration,l_i,m_i,l_o,m_o,elev_o,az_o,az_straight,distance,height,cost\n")
    else:
        print("No output file specified.")
        

    # arrays for estimated parameters from each integration
    l_est = np.array([args.l_guess])
    m_est = np.array([args.m_guess])
    elev_est = np.array([])
    az_est = np.array([])
    height_est = np.array([])

    k = 0

    divs = []

    for bl, vis in compute_visibilities_gen(args.tbn_filename, args.target_freq):
        
        # extract the baseline measurements from the baseline object pairs
        bl2d = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in bl])
        u = bl2d[:, 0]
        v = bl2d[:, 1]

        # we're only fitting the phase, so normalize the visibilities
        vis = vis/np.abs(vis)

        # start the optimization at the mean point of the 10 most recent fits
        l_init = l_est[-10:].mean()
        m_init = m_est[-10:].mean()

        print("Optimizing")
        opt_result = least_squares(
                residual_cplx,
                [l_init, m_init], 
                args=(u, v, vis),
                method='lm'
                )

        print("Optimization result: {}".format(opt_result))
        print("Start point: {}".format((l_init, m_init)))
        l_out, m_out = opt_result['x']
        cost = opt_result['cost']

        l_est = np.append(l_est, l_out)
        m_est = np.append(m_est, m_out)

        elev, az = lm_to_ea(l_out, m_out)

        # use the flat mirror ionosphere model for now
        # TODO: tilted mirror model?
        height = distance * np.tan(elev)/2.0

        csvf.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
            k,
            l_init,
            m_init,
            l_out,
            m_out,
            elev,
            az,
            bearing,
            distance,
            height,
            cost))

        if k in args.scatter:
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

            fig.write_html("scatter_int_{}.html".format(k))

        k += 1
        print("\n\n")

    csvf.close()

    ## remove our initial guess
    #l_est = l_est[1:]
    #m_est = m_est[1:]

    #elev, az = lm_to_ea(l_est, m_est)

    #height = distance * np.tan(elev) / 2.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute visibilities and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('--csv', '-c', type=str,
            help='name of output CSV file')
    parser.add_argument('target_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('l_guess', type=float,
            help='initial guess for l parameter')
    parser.add_argument('m_guess', type=float,
            help='initial guess for m parameter')
    parser.add_argument('scatter', type=int, nargs='*',
            help='export scatter plots for these integrations')
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
