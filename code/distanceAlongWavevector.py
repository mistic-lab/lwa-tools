#!/usr/bin/python

import numpy as np
import argparse
import math
import matplotlib.pyplot as plt

from lsl.common import stations, metabundle, metabundleADP
import known_transmitters
import load_lwa_station
from lwa_common import get_bearing


from IPython import embed

def main(args):

    transmitter_coords = known_transmitters.parse_args(args)

    if transmitter_coords == None:
        print("No transmitter selected.")
        return

    station = load_lwa_station.parse_args(args)
    stands = set(station.getStands()) # convert to set to remove duplicates

    # get transmitter unit vector
    tx_rad = get_bearing(
            [math.degrees(station.lat), math.degrees(station.long)], transmitter_coords
            )

    # note: bearing is measured CW from north
    tx_unit = np.array([-np.cos(np.pi/2 - tx_rad), -np.sin(np.pi/2 - tx_rad)])
    

    ref_stand = next(s for s in stands if s.id in args.reference_stand)
    sec_stands = [s for s in stands if s.id in args.secondary_stands]

    # plot the line along the unit vector
    plt.plot([-100*tx_unit[0], 100*tx_unit[0]], [-100*tx_unit[1], 100*tx_unit[1]])

    print("Reference Stand (id {0}): x = {1}, y = {2}, z = {3}".format(
        ref_stand.id,
        ref_stand.x,
        ref_stand.y,
        ref_stand.z
        ))

    plt.scatter(ref_stand.x, ref_stand.y)


    for s in sec_stands:
        position = (s.x, s.y, s.z)
        print("Secondary Stand (id {0}) position: ({1}, {2}, {3})".format(
            s.id,
            s.x,
            s.y,
            s.z
            ))

        # the vector from the ref stand to s
        diff_vec = np.array([s.x - ref_stand.x, s.y - ref_stand.y])

        # project diff_vec onto the tx unit vector
        distance = abs(np.dot(diff_vec, tx_unit))
        if args.wavelength is not None:
            print(u"\tDistance from reference: {0:.3f}m = {1:.3f}\u03BB".format(distance, distance/args.wavelength))
        else:
            print(u"\tDistance from reference: {0:.3f}m".format(distance))

        plt.scatter(s.x, s.y)


    plt.show()

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Find the distances between stands along the wavevector',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('reference_stand', type=int, nargs=1,
            help='reference (zero-phase) stand ID')
    parser.add_argument('secondary_stands', type=int, nargs='+',
            help='secondary stand IDs (one or more required)')
    parser.add_argument('-l', '--wavelength', type=float,
            help='signal wavelength in meters for relative measurements')
    load_lwa_station.add_args(parser)
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)