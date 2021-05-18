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
