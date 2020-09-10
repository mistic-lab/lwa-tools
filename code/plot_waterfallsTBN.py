from parseTBN import extract_single_ant, pull_meta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse


def main(args):
    directory = args.directory
    dp_stand = args.dp_pol[0]
    pol = args.dp_pol[1]
    t_len = args.num_sec
    
    files = os.listdir(directory)

    fig, ax = plt.subplots(figsize=(20, 15))

    for f in files:

        full_fname=directory+f

        fs = float(pull_meta(full_fname, 'sample_rate'))
        samp_len = t_len * fs
        tStart = pull_meta(full_fname, 'Human start time')
        fc = float(pull_meta(full_fname, 'freq1'))

        arr = extract_single_ant(full_fname, dp_stand, pol, max_length=samp_len)

        fig.suptitle(r"$t_0$ = {}".format(tStart))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        spec, freqs, times, im = ax.specgram(arr, NFFT=2048, Fs=fs, Fc=fc, noverlap=1024)

        plt.savefig('{}.png'.format(f))
        plt.cla()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plots waterfall from start of TBN file.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('directory', type=str, 
        help='directory containing TBN files')
    parser.add_argument('-l', '--num-sec', type=float, 
        help='number of seconds to plot (from start)')
    parser.add_argument('-s', '--dp-pol', default=[10, 1], nargs=2, type=int,  
        help='which dp-stand and polarization to use (default 10, 1)')
args = parser.parse_args()
main(args)