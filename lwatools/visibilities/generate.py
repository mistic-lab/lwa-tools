#!/usr/bin/python3

''' 
This code uses the LSL FX correlator to generate visibilities that can be
used for synthesis imaging.  
'''
import numpy as np

from lsl.correlator import fx as fxc
from lsl.reader.errors import EOFError
from lsl.common import stations
from lsl.correlator import uvutils

from lwatools.visibilities.baselines import uvw_from_antenna_pairs
from lwatools.file_tools.parseTBN import compute_integration_numbers

def extract_tbn_metadata(data_file, antennas, integration_length, verbose=False):
    sample_rate = data_file.get_info('sample_rate')
    center_freq = data_file.get_info('freq1')
    n_samples = data_file.get_info('nframe') / len(antennas)
    samples_per_integration = int(integration_length * sample_rate / 512)
    n_integrations, _ = compute_integration_numbers(data_file, integration_length)

    if verbose:
        print("| Sample rate: {}".format(sample_rate))
        print("| Center frequency: {}".format(center_freq))
        print("| Samples in file: {}".format(n_samples))
        print("| Samples per integration: {}".format(samples_per_integration))
        print("| Integrations in file: {}".format(n_integrations))

    return (sample_rate, center_freq, n_samples, samples_per_integration, n_integrations)


def compute_visibilities(tbn_file, ants, target_freq, station=stations.lwasv, integration_length=1, fft_length=16, use_pol=0, use_pfb=False, verbose=False):
    '''
    Integrates and correlates a TBN file to create an array of visibilities.

    Parameters:
        - tbn_file: TBN file object opened using lsl.reader.ldp.LWASVDataFile
        - ants: a list of antenna objects that should be used
        - station: LSL station object (default: LWASV)
        - integration_length: each integration is this many seconds long (default: 1)
        - fft_length: length of the FFT used in the FX correlator (default: 16)
        - use_pol: currently only supports 0 (X polarization) and 1 (Y polarization) (default: 0)
        - use_pfb: configures the method that the FX correlator uses  (default: False)
        - verbose: whether to display info level logs (default: False)
    Returns:
        (baseline_pairs, visibilities)
        baseline_pairs is a list of pairs of antenna objects indicating which visibility is from where.
        visibilities is a numpy array of visibility vectors, one for each integration. Visibilities within the vectors correspond to the antenna pairs in baselines.

    '''
    if verbose:
        print("Extracting visibilities")
        print("| Station: {}".format(station))

    sample_rate, center_freq, n_samples, samples_per_integration, n_integrations = extract_tbn_metadata(tbn_file, station.antennas, integration_length, verbose=verbose)

    #sometimes strings are used to indicate polarizations
    pol_string = 'xx' if use_pol == 0 else 'yy'

    n_baselines = len(ants) * (len(ants) - 1) / 2 # thanks gauss

    if verbose: print("\nComputing Visibilities:")

    vis_data = np.zeros((n_integrations, n_baselines, fft_length), dtype=complex)

    for i in range(0, n_integrations):
        if verbose: print("| Integration {}/{}".format(i, n_integrations-1))
        #get one integration length of data
        duration, start_time, data = tbn_file.read(integration_length)

        #only use data form the valid antennas
        data = data[[a.digitizer - 1 for a in ants], :]

        baseline_pairs, freqs, visibilities = fxc.FXMaster(data, ants, LFFT=fft_length, pfb=use_pfb, include_auto=False, verbose=True, sample_rate=sample_rate, central_freq=center_freq, Pol=pol_string, return_baselines=True, gain_correct=True)

        # # we only want the bin nearest to our target frequency
        # target_bin = np.argmin([abs(target_freq - f) for f in freqs])

        # visibilities = visibilities[:, target_bin]

        vis_data[i, :, :] = visibilities

    return (baseline_pairs, vis_data)


def compute_visibilities_gen(tbn_file, ants, station=stations.lwasv, integration_length=1, fft_length=16, use_pol=0, use_pfb=False, include_auto=False, verbose=False):
    '''
    Returns a generator to integrate and correlate a TBN file. Each iteration of the generator returns the baselines and the visibilities for one integration

    Parameters:
        - tbn_file: TBN file object opened using lsl.reader.ldp.LWASVDataFile
        - ants: a list of antenna objects that should be used
        - station: LSL station object (default: LWASV)
        - integration_length: each integration is this many seconds long (default: 1)
        - fft_length: length of the FFT used in the FX correlator (default: 16)
        - use_pol: currently only supports 0 (X polarization) and 1 (Y polarization) (default: 0)
        - use_pfb: configures the method that the FX correlator uses  (default: False)
        - verbose: whether to print info level logs (default: False)
    Returns:
        A generator that yields (baseline_pairs, freqs, visibilities).
        baseline_pairs is a list of pairs of antenna objects indicating which
        visibility is from where.
        freqs is a list of frequency bin centers.
        visibilities is a numpy array of visibility samples corresponding to
        the antenna pairs in baselines for each frequency bin.
    '''

    if verbose:
        print('Generating visibilities')
        print('| Station: {}'.format(station))
    antennas = station.antennas

    sample_rate, center_freq, n_samples, samples_per_integration, n_integrations = extract_tbn_metadata(tbn_file, antennas, integration_length, verbose=verbose)

    #sometimes strings are used to indicate polarizations
    pol_string = 'xx' if use_pol == 0 else 'yy'

    # n_baselines = len(ants) * (len(ants) - 1) / 2

    if verbose: print("\nComputing Visibilities:")

    for i in range(0, n_integrations):
        if verbose: print("| Integration {}/{}".format(i, n_integrations-1))
        # get one integration length of data
        try:
            duration, start_time, data = tbn_file.read(integration_length)
        except EOFError:
            print("Reached the end of the TBN file.")
            print("Looks like we calculated the frame numbers wrong. Oops.")
            return
            
            

        #only use data from the valid antennas
        data = data[[a.digitizer - 1 for a in ants], :]

        # correlate
        baseline_pairs, freqs, visibilities = fxc.FXMaster(data, ants, LFFT=fft_length,
                                                pfb=use_pfb, include_auto=include_auto, verbose=True,
                                                sample_rate=sample_rate, central_freq=center_freq,
                                                Pol=pol_string, return_baselines=True, gain_correct=True)

        yield (baseline_pairs, freqs, visibilities)

    return

def simulate_visibilities_gen(model, model_params, freqs, antennas=stations.lwasv.antennas, pol='XX', noise_sigma=None):
    '''
    Returns a generator which provides simulated visibilities according to a specified model.

    Parameters:
        model: a function that takes as arugments
            - u : a np.array of u coordinates
            - v : a np.array of v coordinates
            - some number of parameters (e.g. l, m)
        and returns an np.array the same size as the u and v coordinate vectors containing the 
        visibility samples from the model at the (u,v) points.

        model_params: a list of tuples, each containing values for the
        scalar parameters of model. Each tuple will be used to call model in a
        subsequent iteration of the generator.

        freqs: a list of frequencies. for now these are just used for baseline
        calculation and not passed into the model. TODO: pass freqs to the model

        ants: a list of lsl antenna objects the baselines of which will be
        used to generate the (u,v) coordinate vectors

    Returns:
        A generator yielding a tuple of (baselines, freqs, visibilities)
            - baselines: a list of pairs of antenna objects with each pair representing a baseline
            - freqs: same as the argument freqs
            - visibilities: a numpy array of visibility samples corresponding
              to the antenna pairs in baselines for each frequency in freqs
        The generator will yield a tuple for each set of parameters in model_params.
    '''
    print("Simulating visibilities")
    print(f"| using model {model.__name__}")
    print(f"| received {len(model_params)} sets of parameters, will emit that many sets of visibilities")

    pol1, pol2 = fxc.pol_to_pols(pol)
    antennas1 = [a for a in antennas if a.pol == pol1]
    antennas2 = [a for a in antennas if a.pol == pol2]

    baseline_indices = uvutils.get_baselines(antennas1, antennas2=antennas2, include_auto=False, indicies=True)
    baselines = []
    for bl in range(len(baseline_indices)):
        baselines.append((antennas1[baseline_indices[bl][0]], antennas2[baseline_indices[bl][1]]))

    for params in model_params:

        visibilities = np.empty((len(baselines), len(freqs)), dtype=np.complex128)

        for k, freq in enumerate(freqs):
            wl = 3e8/freq

            uvw = uvw_from_antenna_pairs(baselines, wl)

            u = uvw[:, 0]
            v = uvw[:, 1]
            w = uvw[:, 2]

            visibilities[:, k] = model(u, v, w, *params)

            if noise_sigma is not None:
                noise = np.random.normal(0, noise_sigma, len(visibilities)) + 1j * np.random.normal(0, noise_sigma, len(visibilities))
                visibilities[:, k] += noise


        yield baselines, freqs, visibilities

    return

def build_acm(bl, vis):
    '''
    Builds the array covariance matrix and antenna position vector out of the
    visibilities returned from the correlator.  NOTE: requires that
    visibilities were generated with include_auto = True

    TODO: not well-tested
    '''
    indices = {}
    xyz = []
    n_ants = 0
    for a,b in bl:
        if a.digitizer not in indices:
            indices[a.digitizer] = n_ants
            xyz.append([a.stand.x, a.stand.y, a.stand.z])
            n_ants += 1
        if b.digitizer not in indices:
            indices[b.digitizer] = n_ants
            xyz.append([b.stand.x, b.stand.y, b.stand.z])
            n_ants += 1

    acm = np.zeros((n_ants, n_ants), dtype=np.complex128)

    for k in range(len(vis)):
        a, b = bl[k]

        acm[indices[a.digitizer], indices[b.digitizer]] = vis[k]
        acm[indices[b.digitizer], indices[a.digitizer]] = vis[k]

    return acm, xyz
