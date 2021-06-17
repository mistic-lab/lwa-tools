from typing import List, Optional, Union, Literal
import pandas as pd
from pathlib import Path
from glob import glob
import os
from collections import defaultdict
import math
import warnings

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line

from lsl.reader.ldp import LWASVDataFile, LWA1DataFile
from lsl.common import stations
from lwatools.utils.known_transmitters import transmitter_is_known



#TODO add ability to pull all tarballs and maybe lwa-tv files and maybe pasi pims files


COLUMNS=[
    'tbn_fname', # str
    'tbn_basepath', # str
    'sdf_fname', # str
    'sdf_basepath', # str
    'tarball_fname', # str (optional)
    'tarball_basepath', # str (optional)
    'tx_type', # str in ['tone', 'morse', 'sweep', 'other']
    'station', # str in ['lwasv', 'lwa1']
    'notes', # str (optional)
    'tx_fc', # float (optional unless tx_type=='tone', in which case it is required)
    'known_transmitter', # (str) 'WWV' or 'SFe' or None
    'corrupted', # bool
    'start', # (pandas datetime object) (inferred from tbn file)
    'obs_fc', # float (inferred from tbn file)
    'duration', # float (seconds, inferred from tbn file)
    'disk_size' # float (bytes, inferred from tbn file)
]

class RecordSet():
    """ Handles the entire set of records """
    def __init__(self, recordset_fname:Optional[str]=None)->None:

        if recordset_fname is not None:
            try:
                self.load(recordset_fname)
                print(f'Loaded existing record set from {recordset_fname}')
            except FileNotFoundError:
                self.loaded_filename = recordset_fname
                self.df = pd.DataFrame(columns=COLUMNS)
                print(f'Creating dataframe and storing desired output filename as {recordset_fname}')
        else:
            self.df = pd.DataFrame(columns=COLUMNS)
            self.loaded_filename = None

        self.saved_filename = None

    def load(self, recordset_fname:str)->None:
        self.df = pd.read_csv(filepath_or_buffer=recordset_fname, sep=',', header=0, names=COLUMNS)
        self.loaded_filename = recordset_fname

    def save(self, record_set_fname:Optional[str]=None)->None:
        if record_set_fname is not None:
            self.df.to_csv(path_or_buf=record_set_fname, columns=COLUMNS, sep=',', header=COLUMNS)
            self.saved_filename = record_set_fname
            print(f'Wrote recordset to {record_set_fname}')
        else:
            try:
                self.df.to_csv(path_or_buf=self.loaded_filename, columns=COLUMNS, sep=',', header=COLUMNS)
                self.saved_filename = self.loaded_filename
                print(f'Wrote recordset to {self.loaded_filename}')
            except ValueError:
                raise ValueError ('Need an argument for output filename')
            except FileNotFoundError:
                raise FileNotFoundError ("loaded_filename somehow doesn't exist. Please pass an argument for output filename.")

    def add_observation(self,
        tbn_path:str, sdf_path:str, tx_type:Literal['tone', 'morse', 'sweep',
        'other'], station: Literal['lwasv', 'lwa1'], known_transmitter:Optional[str]=None,
        notes: Optional[str]=None, tx_fc:Optional[float]=None, corrupted:bool=False,
        tarball_path:Optional[str]=None):
        """
        Adds an observation to the dataframe - requires that a dataframe has been loaded previously
        """


        if tx_type == 'tone' and transmitter_is_known(known_transmitter):
            raise ValueError(f"Unrecognized transmitter name {known_transmitter}. See utils/known_transmitters.py.")

        obs_dict = defaultdict() # handles keyerrors
        
        tbnp = Path(tbn_path).resolve()
        tbn_fname = tbnp.name
        tbn_basepath = str(tbnp.parent) + '/'

        sdfp = Path(sdf_path).resolve()
        sdf_fname = sdfp.name
        sdf_basepath = str(sdfp.parent) + '/'

        if tarball_path is not None:
            tarp = Path(tarball_path).resolve()
            tar_fname = tarp.name
            tar_basepath = str(tarp.parent) + '/'
        else:
            tar_basepath=None
            tar_fname=None

        obs_dict = {
            'tbn_fname': tbn_fname, 'tbn_basepath':tbn_basepath,
            'sdf_fname': sdf_fname, 'sdf_basepath': sdf_basepath,
            'tarball_fname': tar_fname, 'tarball_basepath': tar_basepath,
            'tx_type': tx_type,
            'station':station, 'notes': notes,
            'known_transmitter': known_transmitter,
            'corrupted': corrupted}

        # Load up the tbn file so we can parse the metadata
        if station == 'lwasv':
            LWADF = LWASVDataFile
            station_obj = stations.lwasv
        elif station == 'lwa1':
            LWADF = LWA1DataFile
            station_obj = stations.lwa1

        with LWADF(tbn_path) as tbnf:
            # get start time
            start_time = pd.to_datetime(tbnf.get_info('start_time').utc_datetime)
            obs_dict['start'] = start_time

            # get observation center frequency
            obs_fc = tbnf.get_info('freq1')
            obs_dict['obs_fc'] = obs_fc

            # get duration
            num_frames = tbnf.get_info('nframe')
            num_antennas = len(station_obj.antennas)
            samples_per_frame = 512
            num_samples = num_frames * samples_per_frame / num_antennas
            duration = num_samples / tbnf.get_info('sample_rate')
            obs_dict['duration'] = duration

            # get size on disk
            disk_size = os.path.getsize(tbn_path) # can also use tbnf.size
            obs_dict['disk_size'] = disk_size

        # check whether we should infer tx_fc
        if tx_fc is None and known_transmitter == 'WWV':
            if abs(obs_fc-5e6) <50e3:
                tx_fc = 5e6
            else:
                tx_fc = 10e6
            warnings.warn(f"obs_fc is {obs_fc} and transmitter is {known_transmitter} so inferring that tx_fc is {tx_fc}")
        
        if tx_type == 'tone' and tx_fc is None:
            raise ValueError ('Observation cannot be an unknown tone.')

        obs_dict['tx_fc'] = tx_fc
        
        for key, value in obs_dict.items():
            print(f"{key}: {value} (dtype: {type(value)})")
            
        warnings.warn("The above observation was written to the recordSet in memory. You still need to save to file to persist this addition!")
            

        # add it to the dataframe
        self.df = self.df.append(obs_dict, ignore_index=True)
        print()

    def __str__(self):
        loaded_str = f" loaded from {self.loaded_filename}" if self.loaded_filename else ""
        saved_str = f" saved from {self.saved_filename}" if self.saved_filename else ""
        return f"RecordSet object with {len(self.df)} records{loaded_str}{saved_str}"

    def __repr__(self):
        return str(self)
