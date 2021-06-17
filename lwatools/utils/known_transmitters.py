#!/usr/bin/python3
from sys import exit
from dotenv import dotenv_values
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

'''
Module that stores transmitter locations for use by other utilities.  You can
add argparse arguments using the add_cmdline_args function and then parse them
using parse_args.
'''

COLUMNS = [
    'name', # the transmitter's name
    'lat',  # its latitude in decimal degrees
    'lon'   # its longitude in decimal degrees
]

class TransmitterList:
    def __init__(self, tx_list_filename:Optional[str]=None) -> None:
        if tx_list_filename is not None:
            try:
                self.load(tx_list_filename)
                print(f"Loaded transmitter list from {tx_list_filename}")
            except FileNotFoundError:
                self.loaded_filename = tx_list_filename
                self.df = pd.DataFrame(columns=COLUMNS)
                print(f"{tx_list_filename} not found - initializing empty list with that name")
        else:
            # environment file should be in the module root i.e. lwa-tools/lwatools/.env
            environment_filename = str(Path(__file__).resolve().parent.parent) + '/.env'
            environment = dotenv_values(environment_filename)
            try:
                tx_list_filename = environment['KNOWN_TRANSMITTER_FILE']
            except KeyError:
                raise ValueError(f"Env file {environment_filename} does not contain KNOWN_TRANSMITTER_FILE. Please either create {environment_filename} or add this variable")
            
            n = self.load(tx_list_filename)
            print(f"Loaded {n} known transmitters from {self.loaded_filename}")

        self.saved_filename = None

    def load(self, tx_list_filename:str) -> int:
        self.df = pd.read_csv(filepath_or_buffer=tx_list_filename, sep=',', header=0, names=COLUMNS, index_col='name')
        self.loaded_filename = tx_list_filename
        return len(self.df)

    def save(self, tx_list_filename:Optional[str]=None) -> None:
        if tx_list_filename is not None:
            self.df.to_csv(path_or_buf=tx_list_filename, sep=',')
            self.saved_filename = tx_list_filename
        else:
            try:
                self.df.to_csv(path_or_buf=self.loaded_filename, sep=',')
                self.saved_filename = self.loaded_filename
            except ValueError:
                raise ValueError("Need an argument for output filename")
            except FileNotFoundError:
                raise FileNotFoundError("loaded_filename doesn't exist. Please pass an argument for output filename")

        print(f"Saved list of {len(self.df)} transmitters to {self.saved_filename}")

    def add_transmitter(self, name:str, lat:float, lon:float):
        tx_info = {'lat': lat, 'lon': lon}
        new_row = pd.Series(tx_info, name=name)
        self.df = self.df.append(new_row)
        print(f"name : {name}")
        for key, val in tx_info.items():
            print(f"{key}: {val} (dtype: {type(val)})")

        print("The above transmitter was written to the transmitter list in memory. You still need to save to a file to persist this addtion!")
        print(f"The transmitter list (in memory) now contains {len(self.df)} transmitters")

    def get_transmitter_coords(self, tx_name:str) -> list:
        return [self.df.at[tx_name, 'lat'], self.df.at[tx_name, 'lon']]

    def __str__(self) -> str:
        loaded_str = f" loaded from {self.loaded_filename}" if self.loaded_filename else ""
        saved_str = f" saved from {self.saved_filename}" if self.saved_filename else ""
        return f"TransmitterList object with {len(self.df)} records{loaded_str}{saved_str}"

    def __repr__(self) -> str:
        return str(self)


def add_args(parser) -> None:
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


def parse_args(args, known_transmitter_filename:Optional[str]=None) -> Optional[Tuple[float, float]]:
    '''
    Parses arguments object from argparse and returns either the transmitter's
    coordinates or None if no transmitter was specified.
    '''
    if args.unknown_transmitter:
        return None

    if args.transmitter_coords:
        tx_lat, tx_lon = transmitter_coords
        return tx_lat, tx_lon
    
    tx_list = TransmitterList(known_transmitter_filename)

    if args.transmitter_name:
        return tx_list.get_transmitter_coords(args.transmitter_name)

    if args.known_transmitters:
        print("Known transmitters:")
        print(tx_list.df)
        exit()

    raise RuntimeError("No transmitter name or location provided.")

def transmitter_is_known(tx_name:str, known_transmitter_filename:Optional[str]=None) -> bool:
    '''
    Checks if a transmitter with name tx_name exists in the list of known
    transmitters. By default this list is loaded from the file specified in
    lwa-tools/lwatools/.env, but you can specify another file using
    known_transmitter_filename.
    '''
    tx_list = TransmitterList(known_transmitter_filename)
    
    return tx_name in tx_list.df.index


