#!/usr/bin/python3
import h5py

def copy_attrs(source, target):
    """Copies attributes between h5 files.

    Parameters
    ----------
    source : h5 file handle
                where to copy attributes from
    target : h5 file handle
                where to copy the attributes to

    """
    for key, value in source.attrs.items():
        target.attrs[key] = value


def compute_number_of_integrations(tbnf, int_length_seconds):
    '''
    Given a TBN file and an integration length, this function returns the
    number of integrations in the file.
    '''

    n_samples = tbnf.get_info('nframe') / tbnf.get_info('nantenna') * 512
    frames_per_integration = (int_length_seconds * tbnf.get_info('sample_rate')) // 512
    samples_per_integration = frames_per_integration * 512
    n_integrations = int(n_samples / samples_per_integration) + 1
    return n_integrations
