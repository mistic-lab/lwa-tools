import h5py
from lsl.common import stations

from lwatools.utils import known_transmitters

def setup_results_h5(h5_fname, tbnf, transmitter, tx_freq, valid_ants, n_baselines, fft_len, use_pfb, use_pol, integration_length, opt_method, res_function_name, station=stations.lwasv):
    '''
    Opens the hdf5 file that will be used to store results, initializes
    datasets, and writes metadata.
    '''

    print("Writing output to {}".format(h5_fname))
    h5f = h5py.File(h5_fname, 'w')

    transmitter_coords = known_transmitters.get_transmitter_coords(transmitter[0])

    # write metadata to attributes
    ats = h5f.attrs
    ats['tbn_filename'] = tbnf.filename
    ats['transmitter'] = transmitter
    ats['tx_bearing'], _, ats['tx_distance'] = station.get_pointing_and_distance(transmitter_coords + [0])
    ats['tx_freq'] = tx_freq
    ats['sample_rate'] = tbnf.get_info('sample_rate')
    ats['start_time'] = str(tbnf.get_info('start_time').utc_datetime)
    ats['valid_ants'] = [a.id for a in valid_ants]
    ats['n_baselines'] = n_baselines
    ats['center_freq'] = tbnf.get_info('freq1')
    ats['fft_len'] = fft_len
    ats['use_pfb'] = use_pfb
    ats['use_pol'] = use_pol
    ats['int_length'] = integration_length
    ats['opt_method'] = opt_method
    ats['res_function'] = res_function_name 

    n_samples = tbnf.get_info('nframe') / tbnf.get_info('nantenna')
    samples_per_integration = int(integration_length * tbnf.get_info('sample_rate') / 512)
    n_integrations = n_samples / samples_per_integration
    h5f.create_dataset('l_start', (n_integrations,))
    h5f.create_dataset('m_start', (n_integrations,))
    h5f.create_dataset('l_est', (n_integrations,))
    h5f.create_dataset('m_est', (n_integrations,))
    h5f.create_dataset('elevation', (n_integrations,))
    h5f.create_dataset('azimuth', (n_integrations,))
    h5f.create_dataset('height', (n_integrations,))
    h5f.create_dataset('cost', (n_integrations,))
    h5f.create_dataset('skipped', (n_integrations,), dtype='bool')

    return h5f
