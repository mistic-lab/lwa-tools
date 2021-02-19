import numpy as np
import warnings
from scipy.ndimage import center_of_mass


def lm_to_ea(l, m):
    '''
    Converts direction cosines to elevation and azimuth in radians.
    '''
    azimuth = np.pi/2 - np.arctan(m/l)
    
    elev = np.arccos(np.sqrt(l**2 + m**2))

    return elev, azimuth

def flatmirror_height(elev, tx_dist):
    '''
    Computes the reflection virtual height using a simple flat mirror model.
    Assumes that the reflection occurs half way along the line between the transmitter and receiver.
    '''
    return (tx_dist/2) * np.tan(elev)

def tiltedmirror_height(elev, az, tx_az, tx_dist):
    '''
    Computes the reflection virtual height using a tilted mirror model.
    Assumes the reflection occurs somewhere above the perpendicular bisector of the line between the transmitter and receiver.
    '''
    # TODO: not tested :)
    return (tx_dist/2) * np.tan(elev) / np.cos(az - tx_az)

def get_gimg_max(gridded_image, return_img=False):
    # Plot/extract l/m do some modelling
    # I've largely borrow this from plot_gridded_image
    img = gridded_image.image()
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
        return l, m
    else:
        return l,m, img, extent

def get_gimg_center_of_mass(gridded_image, return_img=False):
    #TODO hasn't been tested
    # Plot/extract l/m do some modelling
    # I've largely borrow this from plot_gridded_image
    img = gridded_image.image()
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
        return l, m
    else:
        return l,m, img, extent
