import numpy as np

def uvw_from_antenna_pairs(bl, wavelength=None):
    '''
    Takes bl, a list of lsl antenna objects like the one returned from
    lwatools.visibilities.generate.compute_visibilities_gen and
    returns baseline coordinate vectors as an np array with shape (len(bl), 3).

    You can extract the individual u, v, and w vectors by

        u = uvw[:, 0]
        v = uvw[:, 1]
        w = uvw[:, 2]

    If wavelength is None, the baselines are returned in meters. If not, the
    baselines are divided by wavelength before being returned.
    '''

    uvw = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y, b[0].stand.z-b[1].stand.z]) for b in bl])

    if wavelength:
        uvw = uvw/wavelength

    return uvw

def drop_antennas_outside_radius(antennas, radius):
    '''
    Takes a list of LSL antenna elements like the one in
    lsl.common.stations.lwasv.antennas and filters out all antennas greater
    than radius (in meters) away from the center of the array.
    '''
    sq_r = radius**2

    filtered = [a for a in antennas if (a.stand.x**2 + a.stand.y**2) < sq_r]

    if not filtered:
        raise RuntimeError(f"Radius {radius}m contains no antennas")

    return filtered

def drop_visibilities_outside_radius(bl, vis, radius):
    '''
    Takes the outputs of the visibility generation functions (baseline antenna
    pairs and visibilities) and returns them with contributions from
    antennas outside of the specified radius (in meters) removed.
    '''
    sq_r = radius**2

    both_inside = np.empty(len(bl), dtype=bool)

    bl_filtered = []

    for k,b in enumerate(bl):
        sq_dist0 = b[0].stand.x**2 + b[0].stand.y**2
        sq_dist1 = b[1].stand.x**2 + b[1].stand.y**2

        both_inside[k] = (sq_dist0 < sq_r) and (sq_dist1 < sq_r)

        if both_inside[k]:
            bl_filtered.append(b)

    vis_filtered = vis[both_inside].copy()

    if not bl_filtered:
        raise RuntimeError(f"Radius {radius}m contains no antennas")

    return bl_filtered, vis_filtered
