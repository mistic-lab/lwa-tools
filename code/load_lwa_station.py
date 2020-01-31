#!/usr/bin/python

from lsl.common import stations, metabundle, metabundleADP

'''
Common code that loads LWA info for other utilities. Allows loading built-in
data for LWA1, LWA-SV, or data from an external file. You can add argparse
arguments using the add_cmdline_args function and then parse them using
parse_args.
'''

def add_args(parser):
    parser.add_argument('-s', '--lwasv', action='store_true', 
                        help='use LWA-SV instead of LWA1')
    parser.add_argument('-m', '--metadata', type=str, 
                        help='name of the SSMIF or metadata tarball file to use for mappings')

def parse_args(args):
    '''
    Choose which station object to return
    '''
    if args.metadata is not None:
        try:
            station = stations.parseSSMIF(args.metadata)
        except ValueError:
            try:
                station = metabundle.getStation(args.metadata, ApplySDM=True)
            except:
                station = metabundleADP.getStation(args.metadata, ApplySDM=True)
    elif args.lwasv:
        station = stations.lwasv
    else:
        station = stations.lwa1

    return station

