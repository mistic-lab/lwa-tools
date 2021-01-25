#!/usr/bin/python3

import argparse
import numpy as np
from lsl.common import stations
from lwatools.utils import load_lwa_station

def get_cable_delay(station, standid, pol, fc, verbose=False, fs=None):
    ants = station.antennas
    a = next(a for a in ants if a.stand.id == standid and a.pol == pol)
    delay_s = a.cable.delay(fc)
    delay_rad = delay_s * 2 * np.pi * fc
    if verbose:
        print("Stand {} has cable delay {}s = {}rad = {} samples".format(a.stand.id, delay_s, delay_rad, delay_s*fs))
    return delay_rad

def main(args):
    # get antenna info for the selected station
    station = load_lwa_station.parse_args(args)
    antennas = station.antennas 
    # choose the right antenna
    a = next(a for a in antennas if a.id == args.antenna_id)
    # cable delay is in seconds so we must convert to radians
    print(a.cable.delay(args.frequency) * 2 * np.pi * args.frequency)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='fetches cable delay in radians from LSL',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('frequency', type=float,
            help='signal frequency')
    parser.add_argument('antenna_id', type=int,
            help='stand of interest')
    load_lwa_station.add_args(parser) 
    args = parser.parse_args()
    main(args)

