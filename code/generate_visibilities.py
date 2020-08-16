#!/usr/bin/python2

''' 
This script uses the LSL FX correlator to generate visibilities that can be
used for synthesis imaging.  
'''
import numpy as np
import matplotlib.pyplot as plt

from lsl.correlator import fx as fxc
from lsl.reader.ldp import LWASVDataFile
from lsl.common import stations
from lsl.correlator import uvUtils

########################
# To become parameters:
tbn_filename = "../data/058846_00123426_s0020.tbn"
station = stations.lwasv
integration_length = 1 # in seconds
fft_length = 16
use_pol = 0 # could extend to crosspol later?
use_pfb = False
target_freq = 5351500 # Hz

########################

def extract_tbn_metadata(data_file, antennas):
    sample_rate = data_file.getInfo('sampleRate')
    print("| Sample rate: {}".format(sample_rate))
    center_freq = data_file.getInfo('freq1')
    print("| Center frequency: {}".format(center_freq))

    n_samples = data_file.getInfo('nFrames') / len(antennas)
    print("| Samples in file: {}".format(n_samples))
    samples_per_integration = int(integration_length * sample_rate / 512)
    print("| Samples per integration: {}".format(samples_per_integration))
    n_integrations = n_samples / samples_per_integration
    print("| Integrations in file: {}".format(n_integrations))

    return (sample_rate, center_freq, n_samples, samples_per_integration, n_integrations)

def select_antennas(antennas):
    print("\n\nFiltering for operational antennas:")
    valid_ants = []
    for a in antennas:
        if a.pol != use_pol:
            continue

        if a.getStatus() != 33:
            print("| Antenna {} (stand {}, pol {}) has status {}".format(a.id, a.stand.id, a.pol, a.getStatus()))
            continue

        if a.stand.id == 256:
            print("| Skipping outrigger")
            continue

        valid_ants.append(a)

    print("|=> Using {}/{} antennas".format(len(valid_ants), len(antennas)/2))

    n_baselines = len(valid_ants) * (len(valid_ants) - 1) / 2 # thanks gauss

    return valid_ants, n_baselines


def compute_visibilities(tbn_filename, target_freq, station=stations.lwasv, integration_length=1, fft_length=16, use_pol=0, use_pfb=False):
    '''
    Integrates and correlates a TBN file to create an array of visibilities.

    Parameters:
        - tbn_filename: path of TBN data file
        - station: LSL station object (default: LWASV)
        - integration_length: each integration is this many seconds long (default: 1)
        - fft_length: length of the FFT used in the FX correlator (default: 16)
        - use_pol: currently only supports 0 (X polarization) and 1 (Y polarization) (default: 0)
        - use_pfb: configures the method that the FX correlator uses  (default: False)
    Returns:
        (baseline_pairs, visibilities)
        baseline_pairs is a list of pairs of antenna objects indicating which visibility is from where.
        visibilities is a numpy array of visibility vectors, one for each integration. Visibilities within the vectors correspond to the antenna pairs in baselines.

    '''
    print("Extracting visibilities from file {}:".format(tbn_filename))
    print("| Station: {}".format(station))
    antennas = station.getAntennas()

    data_file = LWASVDataFile(tbn_filename)
    sample_rate, center_freq, n_samples, samples_per_integration, n_integrations = extract_tbn_metadata(data_file, antennas)

    #sometimes strings are used to indicate polarizations
    pol_string = 'xx' if use_pol == 0 else 'yy'

    # some stands may not be operational - we don't want to use those ones
    valid_ants, n_baselines = select_antennas(antennas)

    print("\nComputing Visibilities:")

    vis_data = np.zeros((n_integrations, n_baselines), dtype=complex)

    for i in range(0, n_integrations):
        print("| Integration {}/{}".format(i, n_integrations-1))
        #get one integration length of data
        duration, start_time, data = data_file.read(integration_length)

        #only use data form the valid antennas
        data = data[[a.digitizer - 1 for a in valid_ants], :]

        baseline_pairs, freqs, visibilities = fxc.FXMaster(data, valid_ants, LFFT=fft_length, pfb=use_pfb, IncludeAuto=False, verbose=True, SampleRate=sample_rate, CentralFreq=center_freq, Pol=pol_string, ReturnBaselines=True, GainCorrect=True)

        # we only want the bin nearest to our target frequency
        target_bin = np.argmin([abs(target_freq - f) for f in freqs])

        visibilities = visibilities[:, target_bin]

        vis_data[i, :] = visibilities

    data_file.close()

    return (baseline_pairs, vis_data)


def compute_visibilities_gen(tbn_filename, target_freq, station=stations.lwasv, integration_length=1, fft_length=16, use_pol=0, use_pfb=False):
    '''
    Returns a generator to integrates and correlates a TBN file. Each iteration of the generator returns the baselines and the visibilities for one integration

    Parameters:
        - tbn_filename: path of TBN data file
        - station: LSL station object (default: LWASV)
        - integration_length: each integration is this many seconds long (default: 1)
        - fft_length: length of the FFT used in the FX correlator (default: 16)
        - use_pol: currently only supports 0 (X polarization) and 1 (Y polarization) (default: 0)
        - use_pfb: configures the method that the FX correlator uses  (default: False)
    Returns:
        A generator that yields (baseline_pairs, visibilities).
        baseline_pairs is a list of pairs of antenna objects indicating which
        visibility is from where.
        visibilities is a numpy array of visibility samples corresponding to
        the antenna pairs in baselines.
    '''

    print('Generating visibilities from file: {}'.format(tbn_filename))
    print('| Station: {}'.format(station))
    antennas = station.getAntennas()

    data_file = LWASVDataFile(tbn_filename)
    sample_rate, center_freq, n_samples, samples_per_integration, n_integrations = extract_tbn_metadata(data_file, antennas)

    #sometimes strings are used to indicate polarizations
    pol_string = 'xx' if use_pol == 0 else 'yy'

    # some stands may not be operational - we don't want to use those ones
    valid_ants, n_baselines = select_antennas(antennas)

    print("\nComputing Visibilities:")

    for i in range(0, n_integrations):
        print("| Integration {}/{}".format(i, n_integrations-1))
        # get one integration length of data
        duration, start_time, data = data_file.read(integration_length)

        #only use data from the valid antennas
        data = data[[a.digitizer - 1 for a in valid_ants], :]

        # correlate
        baseline_pairs, freqs, visibilities = fxc.FXMaster(data, valid_ants, LFFT=fft_length, pfb=use_pfb, IncludeAuto=False, verbose=True, SampleRate=sample_rate, CentralFreq=center_freq, Pol=pol_string, ReturnBaselines=True, GainCorrect=True)

        # we only want the bin nearest to our frequency
        target_bin = np.argmin([abs(target_freq - f) for f in freqs])
        
        visibilities = visibilities[:, target_bin]

        yield (baseline_pairs, visibilities)

    data_file.close()
    return
