"""
Script that converts some antennas from a TBN file into hdf5 files.
"""

import os
import numpy as np
import argparse
from datetime import datetime
import h5py
import time
import math

from lsl.reader.ldp import LWASVDataFile

# from parseTBN import get_min_frame_count


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                        length=100, fill=u'\u2588'.encode('utf-8')):
    """
    Call in a loop to create console progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r') # Python 3
    print('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix)), # Python 2 (works but only with the .encode in the def)
    # print('\r--| Progress: %s  %s\r' % ( percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()

# def isFrameLimited(start, length, frames_cap):
#     """Checks whether there's space in an array for another frame. 

#     Parameters
#     ----------
#     start : int
#                 start index where the frame would be written
#     length : int
#                 length of the frame to be written
#     frames_cap : int
#                 frame limit at which to stop writing (frames are of size len(data))

#     Returns
#     -------
#     boolean
#         True if the array will overflow, False if there is space.
#     """
#     num_frames = (start + length) / length
#     if frames_cap < num_frames:
#         return True
#     else:
#         return False




start = time.time()




desired_dp_stands = [141, 178, 180, 224, 225]
mapping = {141: 0, 178: 1, 180: 2, 224: 3, 225: 4}
desired_pol=0
input_file = '/home/nsbruce/projects/def-peterdri/LWA_Data/2019-12-29/RAW_TBN/058846_000123426'
output_file = '/home/nsbruce/temp-NB-JS/058846_000123426_someAnts.h5'
desired_parseIDs = [(141, 0), (178, 0), (180, 0), (224, 0), (225, 0)]

input_filename = '058846_000123426'

num_ants = len(desired_dp_stands)



print("\nChecking input data")
input_data = LWASVDataFile(input_file)

print("\nFinding earliest startTime for all antennas")
IDtimes = {key:0 for key in desired_parseIDs}
donezo = False
while donezo == False:
    current_frame = input_data.readFrame()
    if current_frame.parseID() in desired_parseIDs:
        IDtimes[current_frame.parseID()] = current_frame.getTime()
        if len(set(IDtimes.values()))==1:
            donezo=True
startTime = IDtimes[desired_parseIDs[0]]
print("-| Start Time is: {}".format(startTime))
del IDtimes, donezo

input_data.close() # gotta do this so I don't start partway through

print("\nChecking input data again")
input_data = LWASVDataFile(input_file)


# For getting output array size
total_frames = input_data.getRemainingFrameCount()
# Annoying to do this here
current_frame = input_data.readFrame()
iq_size = len(current_frame.data.iq)
max_possible_length = int(math.ceil( total_frames / input_data.getInfo()['nAntenna'] ) * iq_size)

print("-| Maximum number of frames is: {}".format(max_possible_length))
print("-| Number of antennas to be extracted: {}".format(num_ants))

# Shape is the datasize plus 1 for a counter at each element
output_shape = (num_ants, max_possible_length)
pol0_counters = np.zeros(num_ants, dtype=int)
fill_amts = np.zeros(num_ants, dtype=int)

print("-| Shape of each output dataset will be {}".format(output_shape))

print("\nCreating and opening output file")
with h5py.File(output_file, "w") as f:
    
    # Create a group to store everything in, and to attach attributes to
    print("-| Creating parent group {}[{}]".format(output_file, input_filename))
    parent = f.create_group(input_filename)

    # Add attributes to group
    print("-| Adding TBN metadata as attributes to the parent group")
    for key, value in input_data.getInfo().iteritems():
        if key is "tStart":
            parent.attrs["Human tStart"] = str(datetime.utcfromtimestamp(value))
        parent.attrs[key] = value
        print("--| key: {}  | value: {}".format(key, value))
    parent.attrs['Actual tStart'] = startTime
    
    # Create a subdataset for each polarization
    print("-| Creating datasets full of zeros")
    pol0 = parent.create_dataset("pol0", output_shape, maxshape=(num_ants, None), dtype=np.complex64)#, compression='lzf')
    
    # For progress bar
    totalFrames = input_data.getRemainingFrameCount()
    current_iteration = 0

    print("-| Beginning to build output from input")
    while input_data.getRemainingFrameCount() > 0:
        current_iteration += 1
        printProgressBar(current_iteration,totalFrames)
        if current_frame.parseID() in desired_parseIDs and current_frame.getTime() >= startTime:

            (frame_dp_stand_id, frame_ant_polarization) = current_frame.parseID()

            frameData = current_frame.data.iq

            x_index = mapping[frame_dp_stand_id]

            y_index = pol0_counters[x_index]

            if y_index == 0:
                pol0.attrs[str(frame_dp_stand_id)+'_index'] = mapping[frame_dp_stand_id]
            
            # if not isFrameLimited(y_index, len(frameData), min_frames):
            data_start = y_index
            data_end = data_start+len(frameData)

            pol0[x_index, data_start:data_end] = frameData

            pol0_counters[x_index] = data_end

            fill_amts[x_index]+=len(frameData)


        # Get frame for next iteration
        current_frame = input_data.readFrame()

    # Trim down dataset
    min_fill = min(fill_amts)
    output_shape = (num_ants, min_fill)
    pol0.resize(output_shape)



print("\nDONE")

end = time.time()
totalTime = end-start
print("\nThis script ran for {}s = {}min = {}h".format(totalTime, totalTime/60, totalTime/3600))


