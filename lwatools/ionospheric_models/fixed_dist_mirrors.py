import numpy as np

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
    return (tx_dist/2) * np.tan(elev) / np.cos(az - tx_az)

