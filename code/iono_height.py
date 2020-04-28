#!/usr/bin/python3

"""
Uses LWA data containing signals from a known transmitter to estimate effective ionospheric height.
"""

import argparse
import h5py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import subprocess
import known_transmitters

def main(args):
    # Load data from the H5 file
    h5_file = h5py.File(args.data_filename, 'r')

    # This assumes that the filename is the same as the top-level dataset
    # name in the H5 file.
    data_id = args.data_filename.split('/')[-1].split('.')[0]

    if args.pol1:
        pol_dataset = h5_file[data_id]['pol1']
    else:
        pol_dataset = h5_file[data_id]['pol0']

    # Datasets are indexed by digitizer number, which is just the zero-indexed
    # version of the stand ID. 

    ref_signal = pol_dataset[args.ref_stand - 1]
    sec_signals = [pol_dataset[s - 1] for s in args.secondary_stands]

    # Get the center frequency from args and the sampling rate and measurement
    # center frequency after downmixing from the H5 file
    f_c = args.center_freq
    f_s = h5_file[data_id].attrs['sampleRate']
    f_center = h5_file[data_id].attrs['freq1']
    f_c_dm = f_c - f_center # downmixed center freq

    print("Sample rate: {}".format(f_s))
    print(f"Carrier frequency: {f_c}")
    print(f"Downmixed carrier frequency: {f_c_dm}")

    if input("Show raw reference antenna spectrogram? (y/n)") == 'y':
        plt.figure()
        plt.specgram(ref_signal, NFFT=1024, Fs=f_s, Fc = f_center)
        plt.show()

    # Construct bandpass filter by shifting up a lowpass filter
    cutoff = 500 #Hz
    num_taps = 500
    taps = signal.firwin(num_taps, cutoff/f_s)
    shifts = np.array([np.exp(1j * 2 * np.pi * f_c_dm / f_s * n) for n in range(len(taps))])
    taps = taps * shifts

    if input("Show BPF frequency response? (y/n)") == 'y':
        # plot frequency response of the BPF
        plt.figure()
        fig, ax = plt.subplots(1,1)
        w,h = np.abs(signal.freqz(taps, fs=f_s))
        ax.plot(w, 20 * np.log10(abs(h)), 'k')
        ax2 = ax.twinx()
        ax2.plot(w, np.unwrap(np.angle(h)), 'k--')
        plt.show()

    ref_filtered = signal.lfilter(taps, [1], ref_signal)
    secs_filtered = [signal.lfilter(taps, [1], s) for s in sec_signals]
    
    if input("Show spectrogram post-filtering? (y/n)") == 'y':
        plt.figure()
        plt.specgram(ref_filtered, NFFT=1024, Fs=f_s)
        plt.show()

    # LSL tracks cable delays, so we convert those to phase differences and 
    # use that to correct our phase comparisons
    # We can't call LSL directly from python 3 so we use an external script to do it.
    # Antennas are again numbered differently from stands and digitzers :/
    if args.pol1:
        ref_ant_no = args.ref_stand * 2
        sec_ant_nos = [n * 2 for n in args.secondary_stands]
    else:
        ref_ant_no = args.ref_stand * 2 - 1
        sec_ant_nos = [n * 2 - 1 for n in args.secondary_stands]

    cmd_list = ['python2', 'cable_delay.py', str(f_c)]

    if args.lwasv:
        cmd_list = cmd_list + ['-s']

    ref_cable_phase = float(subprocess.check_output(cmd_list + [str(ref_ant_no)]))

    sec_cable_phases = [float(subprocess.check_output(cmd_list + [str(n)])) for 
            n in sec_ant_nos]


    # this is the phase difference introduced by the difference in length between
    # the signal paths of the reference antenna and each of the secondary antennas
    cable_phase_diffs = [ref_cable_phase - scp for scp in sec_cable_phases]

    # To get the incidence angle we need the baseline distance between antennas 
    # along the wavevector. We need LSL for this, so we call an external python 2 script.

    # -r suppresses all output except relative baselines
    cmd_list = ['python2','distanceAlongWavevector.py', '-r']

    if args.lwasv:
        cmd_list.append('-s') # pass along which station we're using

    cmd_list = cmd_list + ['-t'] + args.transmitter # pass along transmitter arg
    cmd_list = cmd_list + ['-l'] + [str(3e8/args.center_freq)] # set wavelength arg
    cmd_list = cmd_list + [str(args.ref_stand)] 
    baselines = []

    for s in args.secondary_stands:
        baselines.append(float(subprocess.check_output(cmd_list + [str(s)])))

    for b,st in zip(baselines, args.secondary_stands):
        if abs(b) > 0.5:
            print(f"Baseline {b} for stand {st} is greater than half a wavelength")
    
    # preallocate array for phase differences
    phase_diffs = np.zeros(np.shape(secs_filtered))

    for i in range(len(sec_signals)):
        phase_lim = 2 * np.pi * baselines[i]
        phase_diffs[i] = np.angle(ref_filtered * np.conj(secs_filtered[i]) 
                * np.exp(-1j * cable_phase_diffs[i]))
        times_phase_exceeds = len([p for p in phase_diffs[i] if abs(p) > phase_lim])
        print(f"Phase exceeds expected bounds in {times_phase_exceeds/len(phase_diffs[i]) * 100:.3f}% of samples for secondary on stand {args.secondary_stands[i]}")

        phase_diffs[i] = signal.lfilter([1] * 1000, [1000], phase_diffs[i])

    if input("Show phase difference plot? (y/n)") == 'y':
        plt.figure()
        for pd, st, b in zip(phase_diffs, args.secondary_stands, baselines):
            l = plt.plot(pd, label=u"{0} ({1:.3f}\u03BB)".format(st, b))
            plt.plot((0, len(pd)), (-2 * np.pi * b, -2 * np.pi * b), '--', color=l[0].get_color())
            plt.plot((0, len(pd)), (2 * np.pi * b, 2 * np.pi * b), '--', color=l[0].get_color())
        plt.legend(loc='upper right')
        plt.xlabel("Sample")
        plt.ylabel("Phase relative to stand {} (radians)".format(args.ref_stand))
        plt.show()

    # calculate angle of arrival
    angles_of_arrival = [np.arccos(pd / (2 * np.pi * b)) for pd, b in zip(phase_diffs, baselines)]

    if input("Show angle of arrival plot? (y/n)") == 'y':
        plt.figure()
        for aoa, st in zip(angles_of_arrival, args.secondary_stands):
            plt.plot(np.degrees(aoa), label=str(st))
        
        plt.xlabel("Sample")
        plt.ylabel("Arrival angle from horizontal")
        plt.legend(loc="upper right")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='determine the angle of arrival of a signal',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('data_filename', type=str,
            help='name of H5 data file')
    parser.add_argument('center_freq', type=float,
            help='transmitter center frequency')
    parser.add_argument('ref_stand', type=int,
            help='reference stand number')
    parser.add_argument('secondary_stands', type=int, nargs='*',
            help='secondary stand numbers')
    parser.add_argument('-p', '--pol1', action='store_true',
            help='use polarization 1')
    parser.add_argument('-s', '--lwasv', action='store_true',
            help='data is from LWASV')
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
