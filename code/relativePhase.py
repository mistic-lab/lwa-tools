'''
Computes relative phase between antennas from LWA data.
'''

import numpy as np
import argparse
import h5py
from scipy import signal
from parseTBN import extract_single_ant, extract_multiple_ants, pull_meta, generate_multiple_ants
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
    
    print("\n Preparing to process data")

    # construct bandpass filter by shifting up a lowpass filter
    cutoff = 500 #Hz
    num_taps = 500
    print("-| Constructing {}-tap filter with {}Hz cutoff".format(num_taps, cutoff))
    taps = signal.firwin(num_taps, cutoff/f_s)
    shifts = np.array([np.exp(1j * 2 * np.pi * f_c_dm / f_s * n) for n in range(len(taps))])
    taps = taps * shifts

    print("-| Loading phase delay data")
    stn = load_lwa_station.parse_args(args)
    cable_phases = np.array([[get_cable_delay(stn, s, args.pol, f_c, verbose=True, fs=f_s)]
            for s in [args.ref_stand] + args.secondary_stands])


    save_filename = "phase_" + args.data_filename.split('/')[-1].split('.')[0]
    save_filename = save_filename + "_r" + str(args.ref_stand) + ".hdf5"
    print("-| Creating output file:".format(save_filename))
    f = h5py.File(save_filename, 'w')

    f.create_group('relative')
    if args.absolute:
        f.create_group('absolute')
        f['absolute'].create_dataset(str(args.ref_stand), (0,), maxshape=(None,), dtype='f4')

    for stand in args.secondary_stands:
        if args.absolute:
            f['absolute'].create_dataset(str(stand), (0,), maxshape=(None,), dtype='f4')
        f['relative'].create_dataset(str(stand), (0,), maxshape=(None,), dtype='f4')


    print("\nFetching signals")

    for n, sigs in enumerate(generate_multiple_ants(args.data_filename, [args.ref_stand] + args.secondary_stands, args.pol, max_length=n_frames, chunk_length=args.chunk_size)):
        print("-| Processing chunk {}".format(n))

        #filter the signals
        if n == 0:
            first_samples = sigs.take([0], axis=1)
            state = signal.lfilter_zi(taps, [1]) * first_samples

        sigs, state = signal.lfilter(taps, [1], sigs, axis=1, zi=state)

        # correct the phases by shifting the according to the cable phases
        if args.no_phase_corr:
            print(" NOT CORRECTING PHASE ")
        else:
            print(" CORRECTING PHASE ")
            sigs = sigs * np.exp(-1j * cable_phases)

        if args.absolute:
            for stand, phase in zip([args.ref_stand] + args.secondary_stands, np.angle(sigs)):
                stand = str(stand)
                shape = f['absolute'][stand].shape
                pl = len(phase)
                f['absolute'][stand].resize((shape[0] + pl,))
                f['absolute'][stand][-pl:] = phase

        phase_diffs = np.angle(sigs[0] * np.conj(sigs[1:]))

        for stand, phase in zip(args.secondary_stands, phase_diffs):
            stand = str(stand)
            shape = f['relative'][stand].shape
            pl = len(phase)
            f['relative'][stand].resize((shape[0] + pl,))
            f['relative'][stand][-pl:] = phase


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
    parser.add_argument('-a', '--absolute', action='store_true',
            help='dump aboslute phases before filtering')
    parser.add_argument('-c', '--chunk_size', type=int,
            help='size of chunks to read from the TBN file', default=2**20)
    parser.add_argument('-n', '--no_phase_corr', action='store_true',
            help="don't corrrect for cable delays")
    load_lwa_station.add_args(parser)
    args = parser.parse_args()
    
    main(args)
