'''
Tools for generating simulated telescope output data in the TBN format.
'''

import numpy as np
from datetime import datetime

from lsl.sim.tbn import SimFrame
from lsl.common import stations
from lsl.common import dp as dp_common

station = stations.lwasv

tbn_filename = 'test.tbn'

fs = 100e3
fc = 10e6

file_len_s = 5
frame_size = 512

f_signal = fc + 1000
signal_amplitude = 10
signal_el = 30 * np.pi/180
signal_az = 50 * np.pi/180

snr_dB = 5
snr_power = 10**(snr_dB/10)
signal_power = signal_amplitude**2
noise_power = signal_power / snr_power
noise_sigma = np.sqrt(noise_power / 2) # half for real half for imag.

gain = 20 # not sure how this gets set but it's what all of our files have

start_timestamp = int(datetime.timestamp(datetime.now()) * dp_common.fS)

n_frames = int(np.ceil(fs * file_len_s / frame_size))
n_samples = n_frames * frame_size
t_arr = np.linspace(0, file_len_s, n_samples)

tbnf = open(tbn_filename, 'wb')

for k, tf in enumerate(t_arr.reshape(t_arr.shape[0]//frame_size, frame_size)):
    print(f"writing frame {k}/{n_frames - 1}")
    tx_signal = signal_amplitude * np.exp(2j*np.pi*(f_signal - fc)*tf)

    # timestamps are relative to a common sample rate for some reason
    time_delta = int(frame_size * dp_common.fS / fs)
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
        source_unit_vector = np.array([np.cos(signal_el) * np.sin(signal_az), np.cos(signal_el) * np.cos(signal_az), np.sin(signal_el)])
        src_time_delay = np.dot(source_unit_vector, a_xyz) / 3e8
        src_phase_shift = np.exp(2j * np.pi * src_time_delay * f_signal)
        frame.data *= src_phase_shift

        # add noise
        noise = np.random.normal(0, noise_sigma, frame_size) + 1j * np.random.normal(0, noise_sigma, frame_size)
        frame.data += noise

        #noise_pwr = np.real(noise * np.conj(noise)).mean()
        #sig_pwr = np.real(tx_signal * np.conj(tx_signal)).mean()
        #print(f"snr: {10*np.log10(sig_pwr) - 10*np.log10(noise_pwr):.2f}dB")

        # write the frame to the file
        frame.write_raw_frame(tbnf)
