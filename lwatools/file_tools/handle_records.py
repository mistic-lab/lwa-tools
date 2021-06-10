from typing import List, Optional, Union, Literal
import pandas as pd
from pathlib import Path
from glob import glob
import os
from collections import defaultdict

from lsl.reader.ldp import LWASVDataFile, LWA1DataFile
from lsl.common import stations
from lsl.utils.known_transmitters import known_transmitter_locations


"""
DF SHAPE
========

Columns:
    tbn name: str
    sdf no.: int
    sdf name: str
    tx type: str (one of 'continuous tone', 'morse', 'sweeps', 'misc')
    notes: str
    duration: int (seconds)
    station: str [one of 'lwasv', 'lwa1']
    obs fc: float (center frequency of the telescope)
    tx fc: float (center frequency of the tx - only for sweeps, tone, maybe misc)
    start date: string
"""

class RecordSet():
    """ Handles the entire set of records """
    def __init__(self, recordset_fname:Optional[str]=None)->None:

        if recordset_fname is not None:
            self.load(recordset_fname)
        else:
            self.df = pd.DataFrame()
            self.loaded_filename = None
        self.saved_filename = None

    def load(self, recordset_fname:str)->None:
        self.df = pd.read_json(path_or_buf=recordset_fname, type='frame')
        self.loaded_filename = recordset_fname

    def save(self, record_set_fname:str)->None:
        self.df.to_json(path_or_buf=record_set_fname)
        self.saved_filename(record_set_fname)

    # def add_base_paths(self, paths:Union[List[str], str])->None:
    #     #TODO this won't be used I don't think
    #     if type(paths) == str:
    #         # ensure path ends with slash
    #         if paths.endswith('/'):
    #             paths+='/'
    #         self.file_paths.append(paths)
    #     elif type(paths) == list:
    #         # ensure path ends with slash
    #         for path in paths:
    #             if path.endswith('/'):
    #                 path+='/'
    #             self.file_paths.append(path)

    def add_observation(self,
        tbn_path:str, sdf_path:str, tx_type:Literal['tone', 'morse', 'sweep',
        'other'], station: Literal['lwasv', 'lwa1'], known_transmitter:Optional[str]=None,
        notes: Optional[str]=None, tx_fc:Optional[float]=None, corrupted:bool=False):
        """
        Adds an observation to the dataframe - requires that a dataframe has been loaded previously
        """

        #TODO wait does this work. How do we initially add observations.
        assert not self.df.empty, "Must load a record set file first!"

        if known_transmitter:
            if known_transmitter not in known_transmitter_locations.keys():
                raise ValueError(f"Unrecognized transmitter name {known_transmitter}. See utils/known_transmitters.py.")

        obs_dict = defaultdict() # handles keyerrors
        
        tbnp = Path(tbn_path)
        tbn_fname = tbnp.name
        tbn_path = str(tbnp.parent) + '/'

        sdfp = Path(tbn_path)
        sdf_fname = sdfp.name
        sdf_path = str(sdfp.parent) + '/'

        obs_dict = {'tbn_fname': tbn_fname, 'tbn_basepath':tbn_path, 'sdf_fname': sdf_fname, 'sdf_basepath': sdf_path, 'tx_type': tx_type,
            'station':station, 'notes': notes, 'tx_fc': tx_fc, 'known_transmitter': known_transmitter, 'corrupted': corrupted}

        # Load up the tbn file so we can parse the metadata
        if station == 'lwasv':
            tbnf = LWASVDataFile(tbn_path)
            station_obj = stations.lwasv
        elif station == 'lwa1':
            tbnf = LWA1DataFile(tbn_path)
            station_obj = stations.lwa1

        # get duration, obs fc, start time from tbn meta

        # get start time
        start_time = str(tbnf.get_info('start_time').utc_datetime)
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

        # add it to the dataframe
        self.df.append(obs_dict)

    # def find_file(self, fname:str)->Path:
    #     """ Returns the a pathlib object for a given filename (must be in self.paths) """

    #     for path in self.paths:
    #         file_str = glob(path+'/**/'+fname, recursive=True)
    #         if len(file_str) > 1:
    #             raise NameError ('Too many files found')