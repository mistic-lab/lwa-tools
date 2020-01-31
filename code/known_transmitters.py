#!/usr/bin/python

'''
Module that stores transmitter locations for use by other utilities.  You can
add argparse arguments using the add_cmdline_args function and then parse them
using parse_args.
'''

known_transmitter_locations = {
        'WWV' : [40.679917, -105.040944]
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
    parser.add_argument('-t', '--transmitter', nargs='+', metavar=('LAT or NAME', 'LONG'),
                        help='transmitter coordinates in decimal degrees OR provide the name of a known transmitter')
    parser.add_argument('-k', '--known-transmitters', action='store_true',
                        help='list known transmitter names that can be passed with -t')


def parse_args(args):
    '''
    Parses arguments object from argparse and returns either the transmitter's
    coordinates or None if no transmitter was specified.
    '''
    if args.known_transmitters:
        for key in known_transmitter_locations.keys():
            print(key + ": " + str(known_transmitter_locations[key]))
        return None

    if args.transmitter is not None and str(args.transmitter[0]) in known_transmitter_locations:
        return known_transmitter_locations[str(args.transmitter[0])]

    return None


