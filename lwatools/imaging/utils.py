import numpy as np
import warnings
from scipy.ndimage import center_of_mass

from lsl.imaging.data import VisibilityDataSet, PolarizationDataSet
from lsl.imaging.utils import build_gridded_image
from lsl.sim import vis as simVis

from lwatools.utils.array import select_antennas
from lwatools.visibilities.baselines import uvw_from_antenna_pairs


def get_gimg_max(gridded_image, return_img=False, weighting='natural', local_fraction=0.5, robust=0.0):
    # Plot/extract l/m do some modelling
    # I've largely borrow this from plot_gridded_image
    img = gridded_image.image(weighting=weighting, local_fraction=local_fraction, robust=robust)
    imgSize = img.shape[0]
    img = np.roll(img, imgSize//2, axis=0)
    img = np.roll(img, imgSize//2, axis=1)
    l, m = gridded_image.get_LM()
    extent = (m.max(), m.min(), l.min(), l.max())
    l = np.linspace(l.min(), l.max(), img.shape[0])
    m = np.linspace(m.min(), m.max(), img.shape[1])
    if l.shape != m.shape:
        raise RuntimeError("gridded_image is not a square")

    row, col = np.where(img == img.max())
    if len(row) == 1 and len(col) == 1:
        row = row[0]
        col = col[0]
    else:
        warnings.warn("WARNING: Multiple maxes found in this image. Averaging them and returning the result")
        # raise RuntimeError("There are two maxes in this image. This method won't work.")
        row = int(np.mean(row))
        col = int(np.mean(col))

    #! Note the negative
    l = l[-col]
    m = m[row]

    if return_img==False:
        return (l, m)
    else:
        return (l, m, img, extent)

def get_gimg_center_of_mass(gridded_image, return_img=False, weighting='natural', local_fraction=0.5, robust=0.0):
    # Plot/extract l/m do some modelling
    # I've largely borrow this from plot_gridded_image
    img = gridded_image.image(weighting=weighting, local_fraction=local_fraction, robust=robust)
    imgSize = img.shape[0]
    img = np.roll(img, imgSize//2, axis=0)
    img = np.roll(img, imgSize//2, axis=1)
    l, m = gridded_image.get_LM()
    extent = (m.max(), m.min(), l.min(), l.max())
    l = np.linspace(l.min(), l.max(), img.shape[0])
    m = np.linspace(m.min(), m.max(), img.shape[1])
    if l.shape != m.shape:
        raise RuntimeError("gridded_image is not a square")

    row, col = center_of_mass(img)
    row = int(row)
    col=int(col)

    #! Note the negative
    l = l[-col]
    m = m[row]

    if return_img==False:
        return (l, m)
    else:
        return (l, m, img, extent)

def grid_visibilities(bl, freqs, vis, tx_freq, station, valid_ants=None, size=80, res=0.5, wres=0.10, use_pol=0, jd=None):
    '''
    Resamples the baseline-sampled visibilities on to a regular grid. 

    arguments:
    bl = pairs of antenna objects representing baselines (list)
    freqs = frequency channels for which we have correlations (list)
    vis = visibility samples corresponding to the baselines (numpy array)
    tx_freq = the frequency of the signal we want to locate
    valid_ants = which antennas we actually want to use (list)
    station = lsl station object - usually stations.lwasv
    according to LSL docstring:
        size = number of wavelengths which the UV matrix spans (this 
        determines the image resolution).
        res = resolution of the UV matrix (determines image field of view).
        wres: the gridding resolution of sqrt(w) when projecting to w=0.

    use_pol = which polarization to use (only 0 is supported right now)
    returns:
    gridded_image
    '''
    # In order to do the gridding, we need to build a VisibilityDataSet using
    # lsl.imaging.data.VisibilityDataSet. We have to build a bunch of stuff to
    # pass to its constructor.
    
    if valid_ants is None:
        valid_ants, n_baselines = select_antennas(station.antennas, use_pol)

    # we only want the bin nearest to our frequency
    target_bin = np.argmin([abs(tx_freq - f) for f in freqs])

    # Build antenna array
    freqs = np.array(freqs)
    antenna_array = simVis.build_sim_array(station, valid_ants, freqs/1e9, jd=jd, force_flat=True)

    uvw = np.empty((len(bl), 3, len(freqs)))

    for i, f in enumerate(freqs):
        # wavelength = 3e8/f # TODO this should be fixed. What is currently happening is not true. Well it is, but only if you're looking for a specific transmitter frequency. Which I guess we are. I just mean it's not generalized.
        wavelength = 3e8/tx_freq
        uvw[:,:,i] = uvw_from_antenna_pairs(bl, wavelength=wavelength)


    dataSet = VisibilityDataSet(jd=jd, freq=freqs, baselines=bl, uvw=uvw, antennarray=antenna_array)
    if use_pol == 0:
        pol_string = 'XX'
    else:
        raise RuntimeError("Only pol. XX supported right now.")
    polDataSet = PolarizationDataSet(pol_string, data=vis)
    dataSet.append(polDataSet)


    # Use lsl.imaging.utils.build_gridded_image (takes a VisibilityDataSet)
    gridded_image = build_gridded_image(dataSet, pol=pol_string, chan=target_bin, size=size, res=res, wres=wres)

    return gridded_image
