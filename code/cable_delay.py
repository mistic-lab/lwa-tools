#!/usr/bin/python2

import argparse
import numpy as np
from lsl.common import stations
import load_lwa_station

def main(args):
    # get antenna info for the selected station
    station = load_lwa_station.parse_args(args)
    antennas = station.getAntennas() 
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

