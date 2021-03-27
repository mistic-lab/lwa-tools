'''
Tools for generating simulated telescope output data in the TBN format.
'''

import numpy as np
from datetime import datetime

from lsl.sim.tbn import SimFrame
from lsl.common import stations
from lsl.common import dp as dp_common

station = stations.lwasv

tbn_filename = 'test.tbn'

fs = 100e3
fc = 10e6
f_signal_1 = 1000 # relative to fc
frame_size = 512

gain = 20 # not sure how this gets set but it's what all of our files have

file_len_s = 5

start_timestamp = int(datetime.timestamp(datetime.now()) * dp_common.fS)

n_frames = int(np.ceil(fs * file_len_s / frame_size))
n_samples = n_frames * frame_size
t_arr = np.linspace(0, file_len_s, n_samples)

tbnf = open(tbn_filename, 'wb')

for k, tf in enumerate(t_arr.reshape(t_arr.shape[0]//frame_size, frame_size)):
    print(f"writing frame {k}/{n_frames - 1}")
    data = 20*np.exp(2j*np.pi*f_signal_1*tf)

    # timestamps are relative to a common sample rate for some reason
    time_delta = int(frame_size * dp_common.fS / fs)
    timestamp = start_timestamp + k * time_delta

    for a in station.antennas:
        stand = a.stand.id
        pol = a.pol

        frame = SimFrame(stand, pol, fc, gain, k, timestamp, data)

        frame.write_raw_frame(tbnf)
