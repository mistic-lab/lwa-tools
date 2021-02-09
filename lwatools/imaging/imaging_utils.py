import numpy as np


def lm_to_ea(l, m):
    azimuth = np.pi/2 - np.arctan(m/l)
    
    elev = np.arccos(np.sqrt(l**2 + m**2))

    return elev, azimuth

def flatmirror_height(elev, dist):
    return (dist/2) * np.tan(elev)

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
            raise RuntimeError("There are two maxes in this image. This method won't work.")

        #! Note the negative
        l = l[-col]
        m = m[row]

        if return_img==False:
            return l, m
        else:
            return l,m, img, extent
