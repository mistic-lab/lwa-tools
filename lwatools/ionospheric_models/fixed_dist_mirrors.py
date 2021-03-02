import numpy as np
from geographiclib.geodesic import Geodesic

def flatmirror_height(tx_coords, rx_coords, elev):
    '''
    Computes the reflection virtual height using a simple flat mirror model.
    Assumes that the reflection occurs half way along the line between the transmitter and receiver.

    Coordinate arguments should be (lat, lon) pairs in decimal degrees
    elev should be in radians
    '''
    tx_lat, tx_lon = tx_coords
    rx_lat, rx_lon = rx_coords

    # solve the inverse problem to find the tx to rx distance
    rx_to_tx = Geodesic.WGS84.Inverse(rx_lat, rx_lon, tx_lat, tx_lon)

    tx_dist = rx_to_tx['s12']

    return (tx_dist/2) * np.tan(elev)

def flatmirror_location(tx_coords, rx_coords):
    '''
    This is an easy one - it's just halfway along the great circle between TX and RX.

    Arguments should be (lat, long) pairs in decimal degrees.
    '''

    tx_lat, tx_lon = tx_coords
    rx_lat, rx_lon = rx_coords

    # point 1 (lat1, lon1, etc) is the receiver
    # point 2 (lat2, lon2, etc) is the transmitter
    # the inverse problem gives us the length and azimuths of the line connecting two given points
    # note that the azimuth that the line points is different at each point
    rx_to_tx = Geodesic.WGS84.Inverse(rx_lat, rx_lon, tx_lat, tx_lon)

    # the direct problem takes a starting point and a line and tells us the endpoint
    # we'll give it the receiver location and the azimuth and distance determined above
    rx_to_midpoint = Geodesic.WGS84.Direct(rx_lat, rx_lon, rx_to_tx['azi1'], rx_to_tx['s12']/2)

    return rx_to_midpoint['lat2'], rx_to_midpoint['lon2']

def tiltedmirror_height(tx_coords, rx_coords, elev, az):
    '''
    Computes the reflection virtual height using a tilted mirror model.
    Assumes the reflection occurs somewhere above the perpendicular bisector of the line between the transmitter and receiver.

    Coordinate arguments should be (lat, lon) pairs in decimal degrees
    elev and az should be in radians
    '''
    tx_lat, tx_lon = tx_coords
    rx_lat, rx_lon = rx_coords

    # solve the inverse problem to find the tx to rx distance and azimuth at the rx
    rx_to_tx = Geodesic.WGS84.Inverse(rx_lat, rx_lon, tx_lat, tx_lon)

    tx_dist = rx_to_tx['s12']
    tx_az = rx_to_tx['azi1'] * np.pi/180

    return (tx_dist/2) * np.tan(elev) / np.cos(az - tx_az)

def tiltedmirror_location(tx_coords, rx_coords, az):
    '''
    Coordinate arguments should be (lat, long) pairs in decimal degrees.

    az should be the azimuth of arrival in radians
    '''

    tx_lat, tx_lon = tx_coords
    rx_lat, rx_lon = rx_coords

    # convert to degrees
    az = az * 180/np.pi

    # get the line between rx and xx
    rx_to_tx = Geodesic.WGS84.Inverse(rx_lat, rx_lon, tx_lat, tx_lon)

    # find the midpoint of the rx to tx line
    rx_to_midpoint = Geodesic.WGS84.Direct(rx_lat, rx_lon, rx_to_tx['azi1'], rx_to_tx['s12']/2,
            outmask = Geodesic.WGS84.STANDARD | Geodesic.WGS84.REDUCEDLENGTH )

    # rx_to_midpoint contains the "reduced length" m12, which indicates how much a perturbation in azimuth results in a perpendicular offset as we travel along the geodesic
    # we know the azimuth perturbation that points towards the reflection point, so we can find how far off of the rx to tx line the reflection is
    # annoyingly m12 is in radians....
    az_offset = az - rx_to_midpoint['azi1']
    perp_distance = rx_to_midpoint['m12'] * np.pi/180 * az_offset
    # now we can use the direct problem again from the midpoint along the perpendicular to the tx_to_rx line to find the reflection point
    midpoint_to_refl = Geodesic.WGS84.Direct(rx_to_midpoint['lat2'], rx_to_midpoint['lon2'], rx_to_midpoint['azi2'] + 90.0, perp_distance)

    return midpoint_to_refl['lat2'], midpoint_to_refl['lon2']
