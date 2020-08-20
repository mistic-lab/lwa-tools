#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from generate_visibilities import compute_visibilities, select_antennas
from known_transmitters import get_transmitter_coords
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile
from plot_visibility_2d import project_baselines
from scipy.signal import find_peaks
from scipy.optimize import least_squares, brute, fmin
from tqdm import tqdm

import lmfit

"""
JS: This is a WIP - I haven't found a reliable way to fit to wrapped phase data :(
"""

#tbn_filename = "../../data/058846_00123426_s0020.tbn"
#target_freq = 5351500
#transmitter_coords = get_transmitter_coords('SFe')

tbn_filename = "../../data/058628_001748318.tbn"
target_freq = 10e6
transmitter_coords = get_transmitter_coords('WWV')

station = stations.lwasv

def LS_no_wrapping(baselines, visibilities):
    """
    This does a basic least-squares fit to the data. It doesn't work
    because the data is wrapped and this model doesn't account for that. :(
    """

    # express baselines as (u,v) vectors
    bl = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in baseline_pairs])

    # test on first visibility
    phases = np.angle(visibilities[0])

    mat = 4 * np.pi * np.matmul(bl.T, bl)

    vec_phi = np.array([[np.dot(bl[:,0], phases)], [np.dot(bl[:,1], phases)]])

    lm = np.linalg.solve(mat, vec_phi)

    #should be the direction cosines locating the point source in the sky
    l = lm[0][0]
    m = lm[1][0]

    return l, m

def hugo_idea(bl, phases):
    N = lambda j, slope : (phases > (slope*bl + 2*np.pi * j - np.pi)) & (phases < (slope*bl + 2*np.pi * j + np.pi))

    slope = -0.2
    print("{}".format(slope))
    preprod_1 = np.dot(bl, phases)
    preprod_2 = np.dot(bl, bl)

    for i in range(1,200):
        print("\n")
        print(i)
        print(slope)
        #plt.figure()
        #plt.scatter(bl, (slope*bl + np.pi) % (2 * np.pi) - np.pi)    
        acc = 0
        #plt.scatter(bl, phases, s=0.1, alpha=0.5, color='black')
        for j in range(-50, 50):
            #plt.scatter(bl[N(j, slope)], phases[N(j,slope)], s=0.1, label="{}".format(j))
            acc += np.sum(j*bl[N(j, slope)])
            print(len(bl[N(j, slope)]))

        slope = (preprod_1 - 2 * np.pi * acc)/preprod_2
    print(slope)
    plt.scatter(bl, (slope*bl + np.pi) % (2 * np.pi) - np.pi)    
    plt.show()

def histogram_slope(bl, phases):
    threshold_stdev = 3
    threshold_distance = 20

    slope = 0

    for phases in np.angle(visibilities):
    
        threshold = 0.1
        near_wraps = (phases < (-np.pi + threshold)) | (phases > (np.pi - threshold))
        near_zero = (phases < (threshold/2)) & (phases > (-threshold/2))
        
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].scatter(bl, phases, s=0.1)
        ax[1].hist(bl[near_zero], 100, color='black', alpha=0.5)
        ax[1].hist(bl[near_wraps], 100, color='red', alpha=0.5)

        num_bins = 1000
        h, bin_edges = np.histogram(bl[near_wraps], num_bins)

        bin_size = bin_edges[1] - bin_edges[0]

        peaks, _ = find_peaks(h, height=h.mean() + threshold_stdev*h.std(), distance=threshold_distance/bin_size) # justify these parameters
        # distance basically limits the maximum frequency we can work with
        # and therefore the lowest angle?

        wrap_locations = (bin_edges[peaks] + bin_edges[peaks + 1])/2

        h, bin_edges = np.histogram(bl[near_zero], num_bins)

        bin_size = bin_edges[1] - bin_edges[0]

        peaks, _ = find_peaks(h, height=h.mean() + threshold_stdev*h.std(), distance=threshold_distance/bin_size)

        zero_locations = (bin_edges[peaks] + bin_edges[peaks + 1])/2

        ax[1].vlines(wrap_locations, *ax[1].get_ylim(), color='red')
        ax[0].vlines(wrap_locations, *ax[0].get_ylim(), color='red')
        ax[1].vlines(zero_locations, *ax[1].get_ylim(), color='black')
        ax[0].vlines(zero_locations, *ax[0].get_ylim(), color='black')

        if wrap_locations.size == 0:
            # no wraps, so we can just fit a line
            print("fit a line")
            pass
        else:
            for k in range(0, min(len(wrap_locations), len(zero_locations))):
                pass
                
            differences = []
            for k in range(0, min(len(wrap_locations), len(zero_locations))):
                differences.append(abs(wrap_locations[k] - zero_locations[k]))

            rise = np.pi
            run = np.mean(differences)
            last_slope = slope
            slope = -rise/run
            print(slope)

        x = np.linspace(*ax[0].get_xlim(), num=500)
        ax[0].plot(x, (slope*x - np.pi) % (2*np.pi) - np.pi, color='orange')

def point_source_visibility_model(u, v, l, m):
    return np.exp(2j * np.pi * (l*u + v*m))

def point_source_visibility_partials(u, v, l, m):
    mc = point_source_visibility_model(u, v, l, m)
    dfdl = 2j * np.pi * u * mc
    dfdm = 2j * np.pi * v * mc

    return dfdl, dfdm

def gradient(params, u, v, vis):
    l = params[0]
    m = params[1]
    dfdl, dfdm = point_source_visibility_partials(u, v, l, m)

    return np.array([np.abs(dfdl), np.abs(dfdm)]).T
    
def residual_cplx(params, u, v, vis):
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model(u, v, l, m)

    return mc.view(float) - vis.view(float)

def residual_abs(params, u, v, vis):
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model(u, v, l, m)

    return np.abs(mc - vis)

def ls_cost(params, u, v, vis):
    r = residual_abs(params, u, v, vis)
    return np.dot(r,r)


def scatter_model_and_data(u, v, l, m, vis):
    # sorry for using a different plotting library here - 
    # matplotlib's 3d scatter is just so unbearably slow for panning around
    data = [
        go.Scatter3d(x=u, y=v, z=np.angle(vis), mode='markers', marker=dict(size=1, color='red')),
        go.Scatter3d(x=u, y=v, z=np.angle(point_source_visibility_model(u, v, l, m)), mode='markers', marker=dict(size=1, color='black'))
        ]

    fig = go.Figure(data=data)

    fig.update_layout(scene=dict(
        xaxis_title='u',
        yaxis_title='v',
        zaxis_title='phase'))
    fig.show()


def cost_function_contour(l_c, m_c, l_width, m_width, u, v, vis, N=50):
    cost = np.zeros((N,N))
    l_range = np.linspace(l_c - l_width/2.0, l_c + l_width/2.0, N)
    print(l_range)
    m_range = np.linspace(m_c - m_width/2.0, m_c + m_width/2.0, N)
    print(m_range)
    for x, l in enumerate(l_range):
        for y, m in enumerate(m_range):
            cost[x,y] = ls_cost([l, m], u, v, vis)


    plt.contourf(l_range, m_range, cost)
    plt.colorbar()
    plt.xlabel("l")
    plt.ylabel("m")
    plt.show()

if __name__ == "__main__":
    plt.close('all')
    ants, n_baselines = select_antennas(station.antennas, use_pol=0)
    dfile = LWASVDataFile(tbn_filename)
    #baselines, visibilities = compute_visibilities(dfile, ants, target_freq)
    dfile.close()

    azimuth = station.getPointingAndDistance(transmitter_coords + [0])[0]

    bl1d = project_baselines(baselines, azimuth)
    phases = np.angle(visibilities[0])
    

    vis = visibilities[0]
    bl2d = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in baselines])

    u = bl2d[:, 0]
    v = bl2d[:, 1]

    vis = vis/np.abs(vis)

    initial_params = (0.008, 0.031)
    #initial_params = (0.07, 0.035)

    l_width = 1
    m_width = 1
    N = 50

    cost = np.zeros((N,N))

    result = least_squares(residual_abs, initial_params, args=(u, v, vis), max_nfev=1000, method='lm')
    out = result['x']
    print(result)
    print(initial_params)

    cost_function_contour(0, 0, 0.04, 0.04, u, v, vis, N=50)
