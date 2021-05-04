#!/usr/bin/python3
from sys import exit

'''
Module that stores transmitter locations for use by other utilities.  You can
add argparse arguments using the add_cmdline_args function and then parse them
using parse_args.
'''

known_transmitter_locations = {
        'WWV' : [40.679917, -105.040944],
        'SFe' : [35.711389, -106.008333]
    }

def get_transmitter_coords(name):
    '''
    Given a transmitter name, this function returns its lat and long. If the
    transmitter isn't found, we return None.
    '''

    if name in known_transmitter_locations.keys():
        return known_transmitter_locations[name]
    
    return None

def add_args(parser):
    '''
    Adds command line arguments for specifying a transmitter and for printing
    the list of known transmitters.
    '''
    parser.add_argument('-tl', '--transmitter-coords', type=float,
            nargs=2, metavar=('TX_LAT', 'TX_LONG'),
            help='transmitter coordinates in decimal degrees')
    parser.add_argument('-tn', '--transmitter-name', type=str, 
            help='name of a known transmitter')
    parser.add_argument('-k', '--known-transmitters', action='store_true',
            help='list known transmitter names that can be passed with -t')
    parser.add_argument('-u', '--unknown-transmitter', action='store_true',
            help='transmitter is unknown')


def parse_args(args):
    '''
    Parses arguments object from argparse and returns either the transmitter's
    coordinates or None if no transmitter was specified.
    '''
    if args.known_transmitters:
        print("Known transmitter locations:")
        for key in known_transmitter_locations.keys():
            print(key + ": " + str(known_transmitter_locations[key]))
        exit()

    if args.transmitter_name:
        return known_transmitter_locations[args.transmitter_name]
    elif args.transmitter_coords:
        return args.transmitter_coords
    elif args.unknown_transmitter:
        return False
    else:
        raise RuntimeError("No transmitter name or location provided.")
