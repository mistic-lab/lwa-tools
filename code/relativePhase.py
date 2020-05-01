'''
Computes relative phase between antennas from LWA data.
'''

import numpy as np
import argparse
import multiprocessing as mp
from scipy import signal
from parseTBN import extract_single_ant, extract_multiple_ants, pull_meta
from cable_delay import get_cable_delay
import load_lwa_station

def main(args):

    print("\nExtracting metadata")
    f_c = args.center_freq # carrier frequency
    f_s = float(pull_meta(args.data_filename, 'sampleRate')) # measurement sample rate
    f_center = float(pull_meta(args.data_filename, 'freq1')) # measurement center freq
    n_frames = int(pull_meta(args.data_filename, 'nFrames')) # number of frames
    f_c_dm = f_c - f_center # downmixed center frequency

    print("-| Sample rate: {}".format(f_s))
    print("-| Carrier frequency: {}".format(f_c))
    print("-| Downmixed carrier frequency: {}".format(f_c_dm))

    print("\nFetching signals")

    # extract the reference signal

    ## Try something faster ?
    #print("-| Extracting reference signal for stand {}".format(args.ref_stand))
    #sigs[0] = extract_single_ant(args.data_filename, args.ref_stand, args.pol, max_length=n_frames)
    #for k,stand in enumerate(args.secondary_stands):
    #    print("-| Extracting secondary signal for stand {}".format(stant))
    #    sigs[k+1] = extract_single_ant(args.data_filename, stand, args.pol, max_lenght=n_frames)
        
    ## this is faster sometimes but also isn't working on cedar most of the time
    #num_workers = min(mp.cpu_count(), len(args.secondary_stands) + 1) 
    #print("-| Allocating pool of {} worker processes.".format(num_workers))
    #p = mp.Pool(num_workers)
    #results = [p.apply_async(extract_single_ant, (args.data_filename, stand, args.pol),{'max_length': n_frames}) for stand in [args.ref_stand] + args.secondary_stands]
    #p.close()
    #p.join()
    #sigs = np.array([s.get() for s in results])

    ## new strategy...
    sigs = extract_multiple_ants(args.data_filename, [args.ref_stand] + args.secondary_stands, args.pol, max_length=n_frames)

    print("\nBandpass filtering")
    # construct bandpass filter by shifting up a lowpass filter
    cutoff = 500 #Hz
    num_taps = 500
    print("-| Constructing {}-tap filter with {}Hz cutoff".format(num_taps, cutoff))
    taps = signal.firwin(num_taps, cutoff/f_s)
    shifts = np.array([np.exp(1j * 2 * np.pi * f_c_dm / f_s * n) for n in range(len(taps))])
    taps = taps * shifts
    
    # filter the signals
    print("-| Filtering")
    sigs = signal.lfilter(taps, [1], sigs, axis=1)

    # need to account for the phase shift introduced by the cable delays
    print("\nCorrecting for cable delays")
    print("-| Loading phase delay data")
    stn = load_lwa_station.parse_args(args)
    ref_cable_phase = get_cable_delay(stn, args.ref_stand, args.pol, f_c)
    sec_cable_phases = [get_cable_delay(stn, s, args.pol, f_c) for s in args.secondary_stands]


    print("-| Computing cable-induced phase differences")
    cable_phase_diffs = np.array([[ref_cable_phase - scp] for scp in sec_cable_phases])

    print("-| Correcting secondary phases")
    # correct the phases by shifting the according to the cable phases 
    sec_sigs_cable_corrected = np.conj(sigs[1:]) * np.exp(-1j * cable_phase_diffs)

    print("\nComputing phase differences")
    # make it all relative to the reference signal
    phase_diffs = np.angle(sec_sigs_cable_corrected * sigs[0])

    print("\nSaving computed differences")
    save_filename = "rel_phase_" + args.data_filename.split('/')[-1].split('.')[0]
    save_filename = save_filename + "_r" + str(args.ref_stand) + ".npz"
    print("-| Target file: {}".format(save_filename))

    with open(save_filename, 'w') as f:
        arg_dict = dict(zip([str(s) for s in args.secondary_stands], phase_diffs))
        print("-| Writing data")
        np.savez(f, **arg_dict)

    print("\nDONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='compute relative phase between antennas from LWA data',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('data_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('center_freq', type=float,
            help='transmitter center frequency')
    parser.add_argument('ref_stand', type=int,
            help='reference stand number')
    parser.add_argument('secondary_stands', type=int, nargs='*',
            help='secondary stand numbers')
    parser.add_argument('-p', '--pol', type=int, choices=[0,1],
            help='polarization to use', default=0)
    load_lwa_station.add_args(parser)
    args = parser.parse_args()
    
    main(args)
