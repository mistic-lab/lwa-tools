#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
from generate_visibilities import compute_visibilities
from known_transmitters import get_transmitter_coords
from lsl.common import stations


tbn_filename = "../../data/058846_00123426_s0020.tbn"
target_freq = 5351500
transmitter_coords = get_transmitter_coords('SFe')
station = stations.lwasv

azimuth = station.getPointingAndDistance(transmitter_coords + [0])[0]

def plot_vis_2d(bl_array, visibilities):
    plt.close('all')
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(bl_array, abs(visibilities[0,:]), color = 'black', marker='.', s=0.1)
    ax[1].scatter(bl_array, np.angle(visibilities[0,:]), color='black', marker='.', s=0.1)
    plt.show()


def plot_unprojected(baseline_pairs, visibilities):
    # this plots magnitude and phase of visibility as a function of radial baseline from the center of the array
    bl = [np.sqrt((b[0].stand.x - b[1].stand.x)**2 + (b[0].stand.y - b[1].stand.y)**2) for b in baseline_pairs]

    plot_vis_2d(bl, visibilities)


def project_baselines(baseline_pairs, azimuth):
    unit = np.array([-np.cos(np.pi/2 - azimuth), -np.sin(np.pi/2 - azimuth)])

    bl = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in baseline_pairs])

    return np.dot(bl, unit)

def plot_projected(baseline_pairs, visibilities, azimuth):
    bl = project_baselines(baseline_pairs, azimuth)

    plot_vis_2d(bl, visibilities)

