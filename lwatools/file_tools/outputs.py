import h5py
import numpy as np
from lsl.common import stations

from lwatools.file_tools.parseTBN import compute_integration_numbers

def build_output_file(h5_fname, tbnf, valid_ants, n_baselines, integration_length, use_pfb=None, use_pol=None,  tx_freq=None, fft_len=None, opt_method=None, vis_model=None, station=stations.lwasv, transmitter_coords=None, heights=False, mpi_comm=None, verbose=False):
    '''
    Opens the hdf5 file that will be used to store results, initializes
    datasets, and writes metadata.
    '''

    if verbose:
        print("Writing output to {}".format(h5_fname))

    if mpi_comm is None:
        h5f = h5py.File(h5_fname, 'w')
    else:
        h5f = h5py.File(h5_fname, 'w', driver='mpio', comm=mpi_comm)

    # write metadata to attributes
    ats = h5f.attrs
    ats['tbn_filename'] = tbnf.filename
    if transmitter_coords:
        ats['transmitter_coords'] = transmitter_coords
        ats['tx_bearing'], _, ats['tx_distance'] = station.get_pointing_and_distance(transmitter_coords + [0])
    else:
        ats['transmitter_coords'] = np.nan
        ats['tx_bearing'] = np.nan
        ats['tx_distance'] = np.nan
    if tx_freq is not None: ats['tx_freq'] = tx_freq
    ats['sample_rate'] = tbnf.get_info('sample_rate')
    ats['start_time'] = str(tbnf.get_info('start_time').utc_datetime)
    ats['valid_ants'] = [a.id for a in valid_ants]
    ats['n_baselines'] = n_baselines
    ats['center_freq'] = tbnf.get_info('freq1')
    if fft_len is not None: ats['fft_len'] = fft_len
    if use_pfb is not None: ats['use_pfb'] = use_pfb
    if use_pol is not None: ats['use_pol'] = use_pol
    if opt_method is not None: ats['opt_method'] = opt_method
    if vis_model is not None: ats['vis_model'] = vis_model

    n_integrations, duration = compute_integration_numbers(tbnf, integration_length)
    ats['int_length'] = duration
    ats['requested_int_length'] = integration_length

    h5f.create_dataset('l_start', (n_integrations,))
    h5f.create_dataset('m_start', (n_integrations,))
    h5f.create_dataset('l_est', (n_integrations,))
    h5f.create_dataset('m_est', (n_integrations,))
    h5f.create_dataset('elevation', (n_integrations,))
    h5f.create_dataset('azimuth', (n_integrations,))
    if heights==True:
        h5f.create_dataset('height', (n_integrations,))
    h5f.create_dataset('cost', (n_integrations,))
    h5f.create_dataset('nfev', (n_integrations,))
    h5f.create_dataset('skipped', (n_integrations,), dtype='bool')
    h5f.create_dataset('snr_est', (n_integrations,))

    return h5f
