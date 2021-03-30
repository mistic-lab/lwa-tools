'''
Tools for generating simulated telescope output data in the TBN format.
'''

import numpy as np
from datetime import datetime

from lsl.sim.tbn import SimFrame
from lsl.common import stations
from lsl.common import dp as dp_common

def simulate_tbn(tbnfh, len_s, fs, fc, f_signal, src_el, src_az, signal_amp, snr_dB, frame_size=512, gain=20, station=stations.lwasv):
    '''
    Generates a simulated TBN file containing a sinusoidal source and noise.

    Arguments:
        tbnfh:       a file handle that can be written to in binary mode
        len_s:      the desired length of the TBN file in seconds - due to framing this will not be exactly right
        fs:         the sample rate of the file (usually 100kHz)
        fc:         the center frequency of the file (sometimes called freq1)
        f_signal:   the frequency of the signal transmitted by the source
        src_el:     the elevation of the source above the horizon in degrees
        src_az:     the azimuth of the source measured clockwise from north in degrees
        signal_amp: the amplitude of the signal at each antenna
        snr_dB:     the power signal to noise ratio in each antenna channel
        frame_size: the number of samples per frame - can cause issues if not set to 512
        gain:       not really sure what this does but it's 20 in all of our actual data
        station:    an LSL station object representing the telescope used to capture the data
    '''

    # figure out the noise variance
    snr_power = 10**(snr_dB/10)
    signal_power = signal_amp**2
    noise_power = signal_power / snr_power
    noise_sigma = np.sqrt(noise_power / 2) # half for real half for imag.

    start_timestamp = int(datetime.timestamp(datetime.now()) * dp_common.fS)

    n_frames = int(np.ceil(fs * len_s / frame_size))
    n_samples = n_frames * frame_size
    t_arr = np.linspace(0, len_s, n_samples)

    # timestamps are relative to a common sample rate for some reason
    time_delta = int(frame_size * dp_common.fS / fs)

    for k, tf in enumerate(t_arr.reshape(t_arr.shape[0]//frame_size, frame_size)):
        print(f"writing frame {k}/{n_frames - 1}")
        tx_signal = signal_amp * np.exp(2j*np.pi*(f_signal - fc)*tf)

        timestamp = start_timestamp + k * time_delta

        for a in station.antennas:
            stand = a.stand.id
            pol = a.pol
            a_xyz = np.array([a.stand.x, a.stand.y, a.stand.z])
            
            frame = SimFrame(stand, pol, fc, gain, k, timestamp)
            frame.data = np.zeros(frame_size, dtype=complex)

            # add signal
            frame.data += tx_signal

            # phase delay due to the antenna's distance from the phase reference plane
            phase_ctr_unit_vector= np.array([0, 0, 1]) # phase center at zenith
            pc_time_delay = np.dot(phase_ctr_unit_vector, a_xyz) / 3e8
            pc_phase_shift = np.exp(2j * np.pi * pc_time_delay * f_signal)
            frame.data *= pc_phase_shift

            # cable delay
            cbl_time_delay = a.cable.delay(frequency=f_signal)
            cbl_phase_shift = np.exp(-2j * np.pi * cbl_time_delay * f_signal)
            frame.data *= cbl_phase_shift
            
            # add the phase delay due to the source's position in the sky
            src_unit_vector = np.array([np.cos(src_el) * np.sin(src_az), np.cos(src_el) * np.cos(src_az), np.sin(src_el)])
            src_time_delay = np.dot(src_unit_vector, a_xyz) / 3e8
            src_phase_shift = np.exp(2j * np.pi * src_time_delay * f_signal)
            frame.data *= src_phase_shift

            # add noise
            noise = np.random.normal(0, noise_sigma, frame_size) + 1j * np.random.normal(0, noise_sigma, frame_size)
            frame.data += noise

            #noise_pwr = np.real(noise * np.conj(noise)).mean()
            #sig_pwr = np.real(tx_signal * np.conj(tx_signal)).mean()
            #print(f"snr: {10*np.log10(sig_pwr) - 10*np.log10(noise_pwr):.2f}dB")

            # write the frame to the file
            frame.write_raw_frame(tbnfh)
