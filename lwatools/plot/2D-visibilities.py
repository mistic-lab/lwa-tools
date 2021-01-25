#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from ../generate_visibilities import compute_visibilities, select_antennas
from ../known_transmitters import get_transmitter_coords
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

##############################
# Turn these into parameters #
##############################

output_dir = "../../model_fitting"

#tbn_filename = "../../data/058846_00123426_s0020.tbn"
#target_freq = 5351500
#transmitter_coords = get_transmitter_coords('SFe')

tbn_filename = "../../data/058628_001748318.tbn"
target_freq = 10e6
transmitter_coords = get_transmitter_coords('WWV')

station = stations.lwasv


def get_vis_indices(id_pairs):
    indices = []
    for p in id_pairs:
        i = (k for k,b in enumerate(baseline_pairs) if b[0].id == p[0] and b[1].id == p[1])
        indices.append(next(i))

    return indices

        

def plot_vis_2d(bl_array, visibilities, output_dir='.'):
    plt.close('all')
    for k, vis in enumerate(visibilities):
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].scatter(bl_array, abs(vis), color = 'black', marker='.', s=0.1)
        ax[0].set_ylabel("Visibility Magnitude")
        ax[0].set_ylim(0, 60000)
        ax[1].scatter(bl_array, np.angle(vis), color='black', marker='.', s=0.1)
        ax[1].set_ylabel("Visibility Phase")
        ax[1].set_xlabel("Projected Baseline")
        plt.savefig("{}/{}.png".format(output_dir, k))


def plot_unprojected(baseline_pairs, visibilities):
    # this plots magnitude and phase of visibility as a function of radial baseline from the center of the array
    bl = [np.sqrt((b[0].stand.x - b[1].stand.x)**2 + (b[0].stand.y - b[1].stand.y)**2) for b in baseline_pairs]

    plot_vis_2d(bl, visibilities)


def project_baselines(baseline_pairs, azimuth):
    # project all of the visibility measurements from the 2D baseline space on to a specific direction
    # imagine looking at the side of the array perpendicular to the direction of incidence
    unit = np.array([-np.cos(np.pi/2 - azimuth), -np.sin(np.pi/2 - azimuth)])

    bl = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in baseline_pairs])

    return np.dot(bl, unit)

def plot_projected(baseline_pairs, visibilities, azimuth, output_dir='.'):
    bl = project_baselines(baseline_pairs, azimuth)

    plot_vis_2d(bl, visibilities, output_dir)

if __name__ == "__main__":
    azimuth = station.getPointingAndDistance(transmitter_coords + [0])[0]
    tbn_file = LWASVDataFile(tbn_filename)
    ants, n_baselines = select_antennas(stations.lwasv.antennas, use_pol=0)
    baseline_pairs, visibilities = compute_visibilities(tbn_file, ants, target_freq)
    plot_projected(baseline_pairs, visibilities, azimuth, output_dir)
