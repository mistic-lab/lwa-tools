#!/usr/bin/python3
"""
Script that converts TBN data into hdf5 files.
"""

import os
import numpy as np
import argparse
from datetime import datetime
import h5py
import time

from lsl.reader.ldp import LWASVDataFile
from lsl.common import stations

from parseTBN import get_min_frame_count

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

def isFrameLimited(start, length, frames_cap):
    """Checks whether there's space in an array for another frame. 

    Parameters
    ----------
    start : int
                start index where the frame would be written
    length : int
                length of the frame to be written
    frames_cap : int
                frame limit at which to stop writing (frames are of size len(data))

    Returns
    -------
    boolean
        True if the array will overflow, False if there is space.
    """
    num_frames = (start + length) / length
    if frames_cap < num_frames:
        return True
    else:
        return False

def main(args):

    start = time.time()

    print("\nCreating filenames and checking input file extension")
    input_file = args.filename


    ext = os.path.splitext(input_file)[-1].lower()
    if ext not in ['', '.tbn']:
        raise Exception("Extension should be .tbn or not exist")
    else:
        input_filename = os.path.split(os.path.splitext(input_file)[-2])[-1]
        output_file = input_filename + '.hdf5'
        print("-| Input file extension is {} (full name: {})".format(ext, input_file))
        print("-| Output file extension is '.hdf5 (full name: {})".format(output_file))

    print("\nChecking input data")
    input_data = LWASVDataFile(input_file)

    # For getting output array size
    lwasv = stations.lwasv
    num_stands = len(lwasv.stands)
    num_ants = num_stands/2
    min_frames = get_min_frame_count(input_file)
    print("-| Minimum number of frames is: {}".format(min_frames))
    print("-| Number of antennas per polarization: {}".format(num_ants))

    # Annoying to do this here
    current_frame = input_data.read_frame()
    iq_size = len(current_frame.data.iq)

    # Shape is the datasize plus 1 for a counter at each element
    # output_shape_with_counter = (num_ants, min_frames * iq_size + 1)
    output_shape = (num_ants, min_frames * iq_size)
    pol0_counters = np.zeros(num_ants, dtype=int)
    pol1_counters = np.zeros(num_ants, dtype=int)
    print("-| Shape of each output dataset will be {}".format(output_shape))

    print("\nCreating and opening output file")
    with h5py.File(output_file, "w") as f:
        
        # Create a group to store everything in, and to attach attributes to
        print("-| Creating parent group {}[{}]".format(output_file, input_filename))
        parent = f.create_group(input_filename)

        # Add attributes to group
        print("-| Adding TBN metadata as attributes to the parent group")
        for key, value in input_data.get_info().items():
            if key is "tStart":
                parent.attrs["Human tStart"] = str(value.utc_datetime)
            parent.attrs[key] = value
            print("--| key: {}  | value: {}".format(key, value))
        
        # Create a subdataset for each polarization
        print("-| Creating datasets full of zeros")
        pol0 = parent.create_dataset("pol0", output_shape, dtype=np.complex64)#, compression='lzf')
        pol1 = parent.create_dataset("pol1", output_shape, dtype=np.complex64)#, compression='lzf')
        
        # For progress bar
        totalFrames = input_data.get_remaining_frame_count()
        current_iteration = 0

        print("-| Beginning to build output from input")
        while input_data.get_remaining_frame_count() > 0:
            current_iteration += 1
            printProgressBar(current_iteration, totalFrames)
            (frame_dp_stand_id, frame_ant_polarization) = current_frame.id

            frameData = current_frame.data.iq

            x_index = frame_dp_stand_id - 1

            if frame_ant_polarization == 0:
                counter = pol0_counters
                dset = pol0
            elif frame_ant_polarization == 1:
                counter = pol1_counters
                dset = pol1

            y_index = counter[x_index]
            
            if not isFrameLimited(y_index, len(frameData), min_frames):
                data_start = y_index
                data_end = data_start+len(frameData)

                dset[x_index, data_start:data_end] = frameData

                counter[x_index] = data_end


            # Get frame for next iteration
            current_frame = input_data.read_frame()

    print("\nDONE")

    end = time.time()
    totalTime = end-start
    print("\nThis script ran for {}s = {}min = {}h".format(totalTime, totalTime/60, totalTime/3600))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='converts TBN file to hdf5 file', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to convert')
    args = parser.parse_args()
    main(args)