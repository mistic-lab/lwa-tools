import numpy as np


# TODO: This is in both vis and image fitting scripts. put it somewhere general.
def lm_to_ea(l, m):
    azimuth = np.pi/2 - np.arctan(m/l)
    
    elev = np.arccos(np.sqrt(l**2 + m**2))

    return elev, azimuth