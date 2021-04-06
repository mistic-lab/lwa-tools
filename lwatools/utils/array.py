"""
Utilities to do with the physical arrays (stations)
"""

def select_antennas(antennas, use_pol, exclude=None):
    """
    If you wish to skip the outrigger use exclude=[256]. You can also choose to skip other antennas in this manner.
    """
    print("\n\nFiltering for operational antennas:")
    valid_ants = []
    for a in antennas:
        if a.pol != use_pol:
            continue

        if a.combined_status != 33:
            print("| Antenna {} (stand {}, pol {}) has status {}".format(a.id, a.stand.id, a.pol, a.combined_status))
            continue

        if exclude is not None and a.stand.id in exclude:
            print("| Skipping antenna {}".format(a.stand.id))
            continue

        valid_ants.append(a)

    print("|=> Using {}/{} antennas\n".format(len(valid_ants), len(antennas)/2))

    n_baselines = len(valid_ants) * (len(valid_ants) - 1) / 2 # thanks gauss

    return valid_ants, n_baselines

def get_cable_delay(station, standid, pol, fc, verbose=False, fs=None):
    ants = station.antennas
    a = next(a for a in ants if a.stand.id == standid and a.pol == pol)
    delay_s = a.cable.delay(fc)
    delay_rad = delay_s * 2 * np.pi * fc
    if verbose:
        print("Stand {} has cable delay {}s = {}rad = {} samples".format(a.stand.id, delay_s, delay_rad, delay_s*fs))
    return delay_rad