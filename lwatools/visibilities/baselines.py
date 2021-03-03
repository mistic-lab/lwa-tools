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
