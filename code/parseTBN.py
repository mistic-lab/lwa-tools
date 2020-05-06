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
import math

from lsl.reader import tbn, errors
from lsl.reader.ldp import LWASVDataFile
from lsl.common import stations

import arrUtils


def generate_multiple_ants(input_file, dp_stand_ids, polarization, chunk_length=2**20, max_length=-1, truncate=True):
    """Generate chunks of data from a list of antennas.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    dp_stand_ids : list
                list of stand ids from 1 to 256 inclusive
    polarization : list
                antenna polarization
    chunk_length : int
                length of each chunk to extract
    truncate : bool
                if the last chunk is shorter than chunk_length, return a short chunk
                otherwise, pad with zeros
    
    Returns
    -------
    numpy array
        array of size (avail frames, bandwidth)
    """

    input_data = LWASVDataFile(input_file)

    total_frames = input_data.getRemainingFrameCount()
    num_ants = input_data.getInfo()['nAntenna']
    samps_per_frame = 512
    max_possible_length = int(math.ceil( total_frames / num_ants ) * samps_per_frame)

    if max_length < 0:
        max_length = max_possible_length

    print("-| {} frames in file".format(total_frames))
    print("-| {} antennas in file".format(num_ants))
    print("-| {} samples per frame".format(samps_per_frame))
    print("--| Extracting from stands {}, pol {}".format(dp_stand_ids, polarization))
    print("--| There are possibly {} samples for each stand".format(max_possible_length))
    print("--| Returning data in chunks of length {}".format(chunk_length))

    # preallocate array to hold the current chunk of data. leave some space for overflow
    chunk_buffer = np.empty((len(dp_stand_ids), int(chunk_length * 2)), dtype=np.complex64)

    done = False
    samples_sent = 0
    file_ended = False
    fill_levels = [0] * len(dp_stand_ids)

    while not done: 
        # fill the chunk buffer
        while any([l < chunk_length for l in fill_levels]):
            # read a frame
            try:
                current_frame = input_data.readFrame()
            except errors.eofError:
                file_ended = True
                break

            current_id = current_frame.parseID()
            for out_idx, stand in enumerate(dp_stand_ids):
                if (stand, polarization) == current_id:
                    # this is the right stand, add to the buffer
                    wr_idx = fill_levels[out_idx]
                    chunk_buffer[out_idx][wr_idx:wr_idx + samps_per_frame] = current_frame.data.iq
                    fill_levels[out_idx] += samps_per_frame
                    break
         
        if samples_sent + chunk_length >= max_length:
            # this is the last chunk
            print("--| Requested number of samples read")
            done = True
            last_chunk_len = max_length - samples_sent
            if not truncate:
                chunk_buffer[:, last_chunk_len:] = 0
            yield chunk_buffer[:, :last_chunk_len]
        elif file_ended:
            # return unfinished chunk
            print("--| Reached end of file")
            min_fill = min(fill_levels)
            done = True
            if not truncate:
                chunk_buffer[:, last_chunk_len:] = 0
            yield chunk_buffer[:, :min_fill]
        else:
            # yield the chunk
            yield chunk_buffer[:, :chunk_length] 
            samples_sent += chunk_length
            for i in range(len(chunk_buffer)):
                # check if there's more than a chunk of samples for any of the stands
                if fill_levels[i] > chunk_length:
                    # copy the extra samples to the start of the buffer
                    overflow_length = fill_levels[i] - chunk_length
                    print(i, overflow_length)
                    chunk_buffer[i][0:overflow_length] = chunk_buffer[i][chunk_length:fill_levels[i]]
                    # start the next read after the extra samples
                    fill_levels[i] = overflow_length
                else:
                    # otherwise we can overwrite the whole buffer
                    fill_levels[i] = 0
    return

def extract_multiple_ants(input_file, dp_stand_ids, polarization, max_length=-1, truncate=True):
    """Extract and combine all data from a list of antenna into an array of numpy arrays.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    dp_stand_ids : list
                list of stand ids from 1 to 256 inclusive
    polarization : list
                antenna polarization
    max_length : int
                length in samples to extract
    truncate : boolean
                discard later frames so all antennas have the same number
    Returns
    -------
    numpy array
        array of size (len(dp_stand_ids), avail frames)
    """

    input_data = LWASVDataFile(input_file)

    total_frames = input_data.getRemainingFrameCount()
    num_ants = input_data.getInfo()['nAntenna']
    samps_per_frame = 512
    max_possible_length = int(math.ceil( total_frames / num_ants ) * samps_per_frame)

    if max_length < 0:
        max_length = max_possible_length

    print("-| {} frames in file".format(total_frames))
    print("-| {} antennas in file".format(num_ants))
    print("-| {} samples per frame".format(samps_per_frame))
    print("--| Extracting from stands {}, pol {}".format(dp_stand_ids, polarization))
    print("--| Attempting to extract {} of a possible {} samples for each stand".format(max_length, max_possible_length))

    # preallocate data array a little bigger than we think the longest signal will be
    output_data = np.zeros((len(dp_stand_ids), int(max_length*1.2) + 1), dtype=np.complex64)

    fill_levels = [0] * len(dp_stand_ids)

    # while input_data.getRemainingFrameCount() > 0:
    while any([l < max_length for l in fill_levels]):
        try:
            current_frame = input_data.readFrame()
        except errors.eofError:
            print("--| EOF reached before maximum length.")
            break

        current_id = current_frame.parseID()

        # check if this frame is one we want
        matching_stand = next((s for s in dp_stand_ids if (s, polarization) == current_id), -1)

        for s in dp_stand_ids:
            if (s, polarization) == current_id:
                out_index = dp_stand_ids.index(matching_stand)

                if fill_levels[out_index] < max_length:
                    wr_idx = fill_levels[out_index]
                    output_data[out_index][wr_idx:wr_idx + samps_per_frame] = current_frame.data.iq
                    fill_levels[out_index] += samps_per_frame
                break

    
    min_fill = min(fill_levels)

    # if the lengths are unequal then truncate long ones
    if truncate and fill_levels.count(min_fill) != len(fill_levels):
        print("--| Truncating lengths from {} to {}".format(fill_levels, min_fill))
        return output_data[:, :min_fill]
    else:
        return output_data[:, :max_length]


def extract_single_ant(input_file, dp_stand_id, polarization, max_length=-1):
    """Extract and combine all data from a single antenna into a numpy array.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    DP_stand_id : int
                stand id from 1 to 256 inclusive
    polarization : int
                antenna polarization
    max_length : int
                length in samples to extract

    Returns
    -------
    numpy array
        array of size (avail frames, bandwidth)
    """

    input_data = LWASVDataFile(input_file)
    output_data = []

    total_frames = input_data.getRemainingFrameCount()
    num_ants = input_data.getInfo()['nAntenna']
    samps_per_frame = 512
    max_possible_length = math.ceil( total_frames / num_ants ) * samps_per_frame

    if max_length < 0:
        max_length = max_possible_length

    print("-| {} frames in file".format(total_frames))
    print("-| {} antennas in file".format(num_ants))
    print("-| {} samples per frame".format(samps_per_frame))
    print("--| Extracting from stand {}, pol {}".format(dp_stand_id, polarization))
    print("--| Extracting {} of a possible {} samples".format(max_length, max_possible_length))

    # while input_data.getRemainingFrameCount() > 0:
    while len(output_data) < max_length:
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
    # simple_name = os.path.realpath(filename.split('/'))[-1]
    # simple_name = simple_name.split('.')[0]
    simple_name = filename

    print("{} TBN Size: {} kB".format(simple_name, os.path.getsize(simple_name)/1024))

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

def pull_meta(filename, key):
    """ Pulls out metadata from a TBN file given a key.

    Possible keys: 'dataBits','FrameSize','Human tStart', 'tStart','nAntenna','sampleRate','nFrames','tStartSamples','freq1','size'

    Parameters
    ----------
    filename : string
                name of the file to pull the meta from
    key : string
                one of the possible keys listed above
    """
    idfN = LWASVDataFile(filename)
    if key == 'Human tStart':
        tbnKey = 'tStart'
    else:
        tbnKey = key
    value = idfN.getInfo()[tbnKey]
    if key == 'Human tStart':
        return str(datetime.utcfromtimestamp(value))
    else:
        return str(value)

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


def extract_single_ant_from_end(input_file, dp_stand_id, polarization, max_length=-1):
    """Extract and combine all data from a single antenna into a numpy array.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    DP_stand_id : int
                stand id from 1 to 256 inclusive
    polarization : int
                antenna polarization
    max_length : int
            length in samples to extract


    Returns
    -------
    numpy array
        array of size (avail frames, bandwidth)
    """

    input_data = LWASVDataFile(input_file)
    output_data = []

    total_frames = input_data.getRemainingFrameCount()
    num_ants = input_data.getInfo()['nAntenna']
    samps_per_frame = 512
    max_possible_length = math.ceil( total_frames / num_ants ) * samps_per_frame

    if max_length < 0:
        max_length = max_possible_length

    print("-| {} frames in file".format(total_frames))
    print("-| {} antennas in file".format(num_ants))
    print("-| {} samples per frame".format(samps_per_frame))
    print("--| Extracting from stand {}, pol {}".format(dp_stand_id, polarization))
    print("--| Extracting {} of a possible {} samples".format(max_length, max_possible_length))

    while input_data.getRemainingFrameCount() > 0:
    # while len(output_data) < max_length:
        current_frame = input_data.readFrame()

        if current_frame.parseID() == (dp_stand_id, polarization):
            for i in range(len(current_frame.data.iq)):
                output_data.append(current_frame.data.iq[i])
            if len(output_data) > max_length:
                del output_data[:samps_per_frame]

    output_data = np.array(output_data)

    return output_data

def extract_single_ant_from_middle(input_file, dp_stand_id, polarization, max_length=-1, tstart=0):
    """Extract and combine all data from a single antenna into a numpy array.

    Parameters
    ----------
    input_file : string
                raw LWA-SV file path
    DP_stand_id : int
                stand id from 1 to 256 inclusive
    polarization : int
                antenna polarization
    max_length : int
            length in samples to extract
    tstart : int
            UTC timestamp (s since epoch)

    Returns
    -------
    numpy array
        array of size (avail frames, bandwidth)
    """

    input_data = LWASVDataFile(input_file)
    output_data = []

    total_frames = input_data.getRemainingFrameCount()
    num_ants = input_data.getInfo()['nAntenna']
    samps_per_frame = 512
    max_possible_length = math.ceil( total_frames / num_ants ) * samps_per_frame

    if max_length < 0:
        max_length = max_possible_length

    print("-| {} frames in file".format(total_frames))
    print("-| {} antennas in file".format(num_ants))
    print("-| {} samples per frame".format(samps_per_frame))
    print("--| Extracting from stand {}, pol {}".format(dp_stand_id, polarization))
    print("--| Extracting {} of a possible {} samples".format(max_length, max_possible_length))

    while len(output_data) < max_length:
    # while input_data.getRemainingFrameCount() > 0:
        current_frame = input_data.readFrame()
        current_tstamp = current_frame.getTime()

        if current_tstamp >= tstart:
            if current_frame.parseID() == (dp_stand_id, polarization):
                for i in range(len(current_frame.data.iq)):
                    output_data.append(current_frame.data.iq[i])

    output_data = np.array(output_data)

    return output_data
