"""
Saves only a small subset of ACM information and appends the antenna coordinates.
Attempts have been made to increase readability for new h5 users.
"""

import h5py
import numpy as np
from h5Utils import copy_attrs


with h5py.File('058628_001748255_ACM.hdf5', 'r') as h5i:
    coords = np.load('../../LWA_Data/antennas.npy')

    with h5py.File('5MHz_20s_pol0_relativephase.h5','w') as h5o:

        # Copy attributes across
        copy_attrs(h5i, h5o)
        # Delete unnecessary attributes
        del h5o.attrs['FrameSize']
        del h5o.attrs['nFrames']
        del h5o.attrs['size']
        del h5o.attrs['tStartSamples']
        h5o.attrs['centerFreq'] = h5o.attrs['freq1']
        del h5o.attrs['freq1']
        h5o.attrs['f1'] = h5i['freqs'][h5i.attrs['fBin']]
        h5o.attrs['nAntenna'] = h5i['pol0'].shape[0]


        

        h5o.create_dataset('coords', data=coords)
        h5o.create_dataset('times', data=h5i['times'])
        h5o.create_dataset('relative_IQ', data=h5i['pol0'])


