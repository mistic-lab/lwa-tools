# IQ stream -> PFB -> DFT -> ACM

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math


############################# PFB #############################
def quick_pfb(series, nfft, navg):
    """Runs a signal through a polyphase filter bank.

    Parameters
    ----------
    series : array
                time series to be run through the polyphase
    nfft : int
                fft size to be used
    navg : int
                number of ffts to integrate over

    Returns
    -------
    numpy array
        shape is (time_samples, nfft)
    """
    sz = nfft * navg
    spec = []
    for k in np.arange(0, len(series), sz):
        x = series[k:k+sz]
        accum = np.zeros(nfft, dtype=x.dtype)
        if len(x) == sz:
            for i in np.arange(0, sz, nfft):
                fft = np.fft.fftshift(np.fft.fft(x[i:i+nfft]*np.hanning(nfft)))
                accum += fft
            accum /= float(navg)
            spec.append(accum)

    # spec = np.rot90(np.array(spec),3)
    spec = np.array(spec)

    return spec


############################# ACM #############################
def quick_ACM(phys_chan_1, phys_chan_2):
    """Runs two waterfalls into an array covariance matrix.

    Parameters
    ----------
    phys_chan_1 : array
                MUST be of shape (tlen, fft_size)

    Returns
    -------
    numpy array
        shape is (2, 2, tlen, fft_size)
    """
    tlen = phys_chan_1.shape[0]
    fft_size = phys_chan_1.shape[1]

    ACM = np.zeros((2, 2, tlen, fft_size), dtype=phys_chan_1.dtype)
    
    for t in range(tlen):
        for k in range(fft_size):
            x_k = np.array((phys_chan_1[t,k],phys_chan_2[t,k]))
            x_k = x_k.reshape(1,-1) # Cast to column vector
            x_k_h = x_k.conj().T
            ACM[:, :, t, k] = x_k * x_k_h

    return ACM


############################# COMPLEX SINE #############################
def make_complex_sine(f, fs, t, phi=0, A=1):
    """Makes complex sine wave.

    Parameters
    ----------
    f : float
                frequency of sine
    fs : float
                sampling rate
    t : float
                length of signal in seconds
    phi : float
                phase offset in radians (default: 0)
    A : float
                amplitude (default: 1)
    Returns
    -------
    numpy array
        sampled sine wave
    """
    t_arr = np.linspace(0,t,t*fs) # time in s
    omega = f * 2 * np.pi
    x = A * np.exp( 1j * ( omega * t_arr + phi ))

    return x


############################# PLOT WATERFALLS #############################
def show_waterfalls(arr, fc, fs, spec_index=0, title=''):
    """Prints waterfalls and spectra for phase and magnitude of a 2D spectrogram array.

    Parameters
    ----------
    arr : array
                2D complex array of spectrogram (must be of shape (tlen, fft_size))
    fc : float
                center frequency
    fs : float
                sampling rate
    spec_index : int
                which index of the array to show the spectrum from
    """
    
    fft_size = arr.shape[1]
    tlen = arr.shape[0]
    freq_arr = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0/fs)) + fc
    t_arr = np.linspace(0,tlen*fs, tlen)

    magHigh = np.ceil(np.max(10.*np.log10(np.abs(arr)))) # for cbar

    magMat = 10.*np.log10(np.abs(arr))
    magSpec = magMat[spec_index]
    phaseMat = np.angle(arr)
    phaseSpec = phaseMat[spec_index]

    fig, axs = plt.subplots(2,2, sharex='row')
    fig.suptitle(title)

    axs[0,0].plot(freq_arr, magSpec)
    axs[0,0].set_title('Magnitude')
    axs[0,0].set_ylabel('dB')

    cmmag = axs[1,0].matshow(magMat)
    axs[1,0].set_ylabel('Time (s)')
    axs[1,0].set_xticks([])
    fig.colorbar(cmmag, label='dB', ax=axs[1, 0], orientation='horizontal')

    axs[0,1].plot(freq_arr, phaseSpec)
    axs[0,1].set_title('Phase')
    axs[0,1].set_ylabel('rads')
    axs[0,1].yaxis.tick_right()
    axs[0,1].yaxis.set_label_position("right")
    # axs[0,1].set_ylim(-np.pi, np.pi)


    cmphase = axs[1,1].matshow(phaseMat)
    axs[1,1].set_xticks([])
    axs[1,1].set_ylabel('Time (s)')
    axs[1,1].yaxis.tick_right()
    axs[1,1].yaxis.set_label_position("right")
    # cmphase.set_clim(vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cmphase, label='rads', ax=axs[1, 1], orientation='horizontal')


    plt.tight_layout()
    plt.show()






############################# MAIN #############################
def main(args):

    ## Setup arguments
    fft_size = args.nfft
    fc = args.fc
    fs = args.fs
    f = args.f1
    tend = args.time
    navg = args.navg



    ## Make IQ Streams
    simple_f = f-fc
    x1 = make_complex_sine(simple_f, fs, tend, A=10)
    x2 = make_complex_sine(simple_f, fs, tend, phi=3.14, A=10)

    # print("max(x1 phase)={}".format(np.amax(np.angle(x1))))
    # print("min(x1 phase)={}".format(np.amin(np.angle(x1))))
    # print("max(x2 phase)={}".format(np.amax(np.angle(x2))))
    # print("min(x2 phase)={}".format(np.amin(np.angle(x2))))
    
    # print("x1 initial phase={}".format(np.angle(x1[0])))
    # print("x2 initial phase={}".format(np.angle(x2[0])))

    # # Plot inputs
    # mul = 5 # how many periods to plot
    # period = simple_f**-1
    # seconds_to_plot = period*mul
    # samples_per_period = math.ceil(period*fs)
    # samples_to_plot = samples_per_period*mul
    # t_arr = np.linspace(0,seconds_to_plot,samples_to_plot) # time in s

    # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # f.suptitle('First {} periods'.format(mul))

    # color='tab:red'
    # ax1.plot(t_arr, np.real(x1[:mul*samples_per_period]), color=color, label='Real', linestyle=':')
    # ax1.plot(t_arr, np.imag(x1[:mul*samples_per_period]), color=color, label='Imag.', linestyle='--')
    # ax1.set_ylabel('Mag.', color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax2.plot(t_arr, np.real(x2[:mul*samples_per_period]), color=color, label='Real', linestyle=':')
    # ax2.plot(t_arr, np.imag(x2[:mul*samples_per_period]), color=color, label='Imag.', linestyle='--')
    # ax2.set_ylabel('Mag.', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax1.legend()
    # ax2.legend()

    # ax3 = ax1.twinx()
    # ax4 = ax2.twinx()

    # color='tab:blue'
    # ax3.plot(t_arr, np.angle(x1[:mul*samples_per_period]))
    # ax3.set_ylabel('Phase', color=color)
    # ax3.tick_params(axis='y', labelcolor=color)
    # # ax3.set_yticks([-3.14,0,3.14])
    # ax4.plot(t_arr, np.angle(x2[:mul*samples_per_period]))
    # ax4.set_ylabel('Phase', color=color)
    # ax4.tick_params(axis='y', labelcolor=color)
    # # ax4.set_yticks([-3.14,0,3.14])

    # ax2.set_xlabel('Time (s)')
    # ax1.set_title('x1')  
    # ax2.set_title('x2')  
    # plt.show()



    ## Run each stream through a PFB/DFT
    x1_spec = quick_pfb(x1, fft_size, navg)
    x2_spec = quick_pfb(x2, fft_size, navg)

    # show_waterfalls(x1_spec, fc=fc, fs=fs, title='x1 after PFB')
    # show_waterfalls(x2_spec, fc=fc, fs=fs, title='x2 after PFB')


    ## ACM those bad boys
    ACM = quick_ACM(x1_spec, x2_spec)

    # show_waterfalls(ACM[0,0], fc=fc, fs=fs, title='ACM[0,0]')
    show_waterfalls(ACM[0,1], fc=fc, fs=fs, title='ACM[0,1]')
    # show_waterfalls(ACM[1,0], fc=fc, fs=fs, title='ACM[1,0]')
    # show_waterfalls(ACM[1,1], fc=fc, fs=fs, title='ACM[1,1]')



    # Waterfalls
    # f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    # f.suptitle('ACM[0,1]')
    # ax1.matshow(10.*np.log10(np.abs(ACM[0,1,:,:])))
    # ax1.set_title('Magnitude')  
    # ax2.matshow(np.angle(ACM[0,1,:,:]))
    # ax2.set_title('Phase')  
    # ax2.set_xlabel('Freqs')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='simulates the small beamformer, using LWA TBN parameters', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-n', '--nfft', type=int, default=512, 
                        help='fft size')
    parser.add_argument('-a', '--navg', type=int, default=10, 
                        help='number of ffts to average (PFB taps)')
    parser.add_argument('-s', '--fs', type=int, default=100e3, 
                        help='sampling frequency in Hz')
    parser.add_argument('-c', '--fc', type=float, default=9.9749999987*10**6, 
                        help='sampling frequency in Hz')
    parser.add_argument('-f', '--f1', type=float, default=10e6,
                        help='frequency to make sine at in Hz')
    parser.add_argument('-t', '--time', type=float, default=10,
                        help='time length in s to simulate')

    args = parser.parse_args()
    main(args)