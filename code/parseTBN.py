##########
# Author: Nicholas Bruce
# Date: June 17 2019
#
# Functions to pre-process LWASV data. LSL required.
#
##########

import os
import numpy as np
import numpy.lib.format as fmt
import urllib2
from datetime import datetime
import h5py

from lsl.reader import tbn
from lsl.reader.ldp import LWASVDataFile
from lsl.common import stations

import arrUtils


def extract_single_ant(input_file, dp_stand_id, polarization):
    """Extract and combine all data from a single antenna into a numpy array.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    DP_stand_id : int
                stand id from 1 to 256 inclusive
    polarization : int
                antenna polarization

    Returns
    -------
    numpy array
        array of size (avail frames, bandwidth)
    """

    input_data = LWASVDataFile(input_file)
    output_data = []

    while input_data.getRemainingFrameCount() > 0:
        current_frame = input_data.readFrame()

        if current_frame.parseID() == (dp_stand_id, polarization):
            for i in range(len(current_frame.data.iq)):
                output_data.append(current_frame.data.iq[i])

    output_data = np.array(output_data)

    return output_data


def meta_to_txt(filename):
    """Pulls metadata from TBN file and puts it into a txt file of the same name

    Parameters
    ----------
    filename : string
                name of file to be read (may end in dat, tbn, or nothing)
    """


    # print localFile
    simple_name = os.path.realpath(filename.split('/'))[-1]
    simple_name = simple_name.split('.')[0]

    print("{} TBN Size: {} kB".format(simple_name, os.path.getsize(filename)/1024))

    # Check that the datatype is correct according to lsl
    idfN = LWASVDataFile(filename)
    print("{} is of type: {}".format(simple_name, type(idfN)))

    # Poll the TBN file for its specifics
    with open(simple_name + ".txt", 'w') as meta:
        meta.write('TBN Metadata:\n')
        for key,value in idfN.getInfo().iteritems():
            if key is "tStart":
                meta.write("Human tStart: " + str(datetime.utcfromtimestamp(value))+"\n")
            meta.write("  %s: %s\n" % (str(key), str(value)))


    idfN.close()

def make_sample_tbn(filename, num_frames=2000000):
    """Takes the defined number of frames and writes them to a new .tbn file

    Parameters
    ----------
    filename : string
                name of file to be read (may end in dat, tbn, or nothing)
    num_frames  :  int
                number of frames to be kept (default: 2000000)
    """

    # Make string for urllib2
    localFile='file:///'+os.path.realpath(filename)

    # print localFile
    simple_name = os.path.realpath(filename).split('/')[-1]
    simple_name = simple_name.split('.')[0]

    # Pull from dat and make tbn file
    if not os.path.exists(simple_name+'.tbn'):
        fh1 = urllib2.urlopen(localFile)
        fh2 = open(simple_name+'.tbn', 'wb')
        fh2.write(fh1.read(tbn.FrameSize*num_frames))
        fh1.close()
        fh2.close()
    print("\n%s TBN Size: %.1f kB".format(simple_name, os.path.getsize(simple_name+'.tbn')/1024.))

    # Check that the datatype is correct according to lsl
    idfN = LWASVDataFile(simple_name+'.tbn')
    print("%s is of type: %s \n".format(simple_name, type(idfN)))


    idfN.close()

def count_frames(filename):
    """Prints out the number of frames for each antenna from a TBN file

    Parameters
    ----------
    filename : string
                name of file to be read (may end in dat, tbn, or nothing)
    """

    def __getKeysByValue__(myDict, valueToFind):
        listOfKeys = []
        listOfItems = myDict.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys.append(item[0])
        return  listOfKeys


    bigDict = {}

    idfN = LWASVDataFile(filename)
    total_num_frames = idfN.getRemainingFrameCount()

    while idfN.getRemainingFrameCount() > 0:
        current_frame = idfN.readFrame()
        key = str(current_frame.parseID())
        
        try:
            bigDict[key] = bigDict[key] + 1
        except KeyError:
            bigDict[key] = 1
    
    # Make a list of unique frame counts
    unique_frame_counts = set(bigDict.values())

    # Create dict with key = num_ants that each have value = num_frames
    antsFramesDict = {}

    for i in unique_frame_counts:
        num_frames = i
        num_ants = len(__getKeysByValue__(bigDict,num_frames))
        antsFramesDict[num_ants] = num_frames
    

    total_calculated_frames = 0

    print("STATS")
    print("-> Total number of frames in file: %s" % total_num_frames)
    for key, value in antsFramesDict.iteritems():
        print("---> Number of antennas with %s frames: %s" %(value, key))
        total_calculated_frames = total_calculated_frames + (key * value)
    print("SANITY CHECK")
    print("-> Frames")
    print("---> Sum of frames = {}".format(total_calculated_frames))
    print("-> Antennas")
    print("---> Sum of antennas = {}".format(sum(antsFramesDict.keys())))


def get_min_frame_count(filename):
    """Prints out smallest frame count of all antennas.

    Parameters
    ----------
    filename : string
                name of file to be read (may end in dat, tbn, or nothing)

    Returns
    -------
    int
        minimum number of frames available
    """

    bigDict = {}

    idfN = LWASVDataFile(filename)

    while idfN.getRemainingFrameCount() > 0:
        current_frame = idfN.readFrame()
        key = str(current_frame.parseID())
        
        try:
            bigDict[key] = bigDict[key] + 1
        except KeyError:
            bigDict[key] = 1

    min_count = min(bigDict.values())

    return min_count



def TBN_to_freq_bin_matrix_indexed_by_dp_stand(filename, Fc, f1, fft_size=512, Fs=100000, polarization=0):
    """Reads each from of a TBN, takes an FFT, and puts a single bin of it into an index
        particular to it's DP stand number. It continues to append bin values as so each index
        is the full time-series of frequency bin values of that DP stand. It concats the vectors
        to be the length of the shortest so that the resulting matrix is rectangular.

        *LIMITATION* : It only does one polarization.

    Parameters
    ----------
    filename : string
                name of file to be read (may end in dat, tbn, or nothing)
    Fc : float
                center frequency in Hz
    f1 : float
                frequency of the signal to extract
    fft_size : int
                size of FFT window
    Fs : int
                sampling rate
    polarization : int
                which polarization to process, either 0 (default) or 1

    Returns
    -------
    numpy array
        array of size (num_dp_stands, samples_in_time_series)
    """

    bin_of_f1 = arrUtils.get_frequency_bin(fc=Fc, f1=f1, fft_size=fft_size)
    input_data = LWASVDataFile(filename)

    lwasv = stations.lwasv
    num_stands = len(lwasv.getStands())
    num_ants = num_stands/2

    #how many frames in total
    frame_count = input_data.getRemainingFrameCount()
    
    num_frames_per_ant = frame_count/num_ants

    # plus 1 to have space for a counter
    output_data = np.zeros((num_stands, num_frames_per_ant+1), dtype=np.complex64)

    current_frame = input_data.readFrame()
    # iq_size = len(current_frame.data.iq)

    count=1

    while input_data.getRemainingFrameCount() > 0:
        (dp_stand_id, ant_polarization) = current_frame.parseID()
        if ant_polarization == polarization:
            #NOT the same thing as the LWA stand number
            index = dp_stand_id - 1

            # Which cell to write to
            count = int(np.real(output_data[index,0]) + 1)

            if count < num_frames_per_ant:
            
                fft=np.fft.fftshift(np.fft.fft(current_frame.data.iq))

                pt = fft[bin_of_f1]
                output_data[index, count] = pt

                # update counter
                output_data[index, 0] = count
        # Get frame for next iteration
        current_frame = input_data.readFrame()

    # Remove counter
    output_data = output_data[:,1:]

    return output_data


def dp_stand_indexed_matrix_to_ant_indexed_matrix(dp_stand_arr):
    """Takes a matrix where the indexes are dp stand numbers and returns a matrix where
        the indexes are antenna numbers

        *LIMITATION* : It only does one polarization.

    Parameters
    ----------
    dp_stand_arr : numpy array
                matrix where indexes are dp stand numbers

    Returns
    -------
    numpy array
        array of size (num_antenna_stands, len(dp_stand_arr[0]))
    """

    lwasv = stations.lwasv

    antennas = lwasv.getAntennas()
    # divide by two because a single polarization
    num_antennas = len(antennas)/2

    ant_arr = np.zeros(dp_stand_arr.shape, dtype = np.complex64)

    for i in range(num_antennas):
        digitizer = 2*(i-1)+1
        ant_stand = antennas[digitizer-1].stand.id

        ant_arr[ant_stand-1,:] = dp_stand_arr[i,:]

    return ant_arr


def write_single_antenna_to_binary_file(input_file, dp_stand_id, polarization, output_file):
    """Extract a single dp_stand/pol to a npy file.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    dp_stand_id : int
                stand id from 1 to 256 inclusive
    polarization : int
                antenna polarization
    output_file : string
                filename to be saved/appended to
    """

    if not output_file.endswith(".singleAnt"):
        output_file = output_file + ".singleAnt"

    input_data = LWASVDataFile(input_file)

    with open(output_file, 'ab') as f:
        while input_data.getRemainingFrameCount() > 0:
            current_frame = input_data.readFrame()
            if current_frame.parseID() == (dp_stand_id, polarization):
                float_arr = np.array(current_frame.data.iq).view(float)
                float_arr.tofile(f)

    
def load_binary_file_to_array(filename):
    """Extract a complex array from a binary file of floats

    Parameters
    ----------
    filename : string
                binary file path/name

    Returns
    -------
    numpy array
        array of unknown size (same datasize as the binary file)
    
    """
    arr = np.fromfile(filename, dtype=np.complex64)
    return arr


