import math
import numpy as np

def get_bearing(latlong_a, latlong_b, output_radians=True):
    """
    Returns bearing counterclockwise from north as if you're looking
    from point A to point B. You can set the output format to degrees by
    passing output_radians=False.
    
    :Parameters:
    - latlong_a: a tuple or list of the lat and long coordinates of point A in decimal degrees
    - latlong_b: a tuple or list of the lat and long coordinates of point B in decimal degrees
    - output_radians: sets the output angular measure

    :Returns:
    - the bearing in degrees or radians
    """

    lat_a = math.radians(latlong_a[0])
    lat_b = math.radians(latlong_b[0])

    dlong = math.radians(latlong_b[1] - latlong_a[1])

    rad_bearing = math.atan2(
            math.sin(dlong) * math.cos(lat_b),
            math.cos(lat_a) * math.sin(lat_b) - (math.sin(lat_a) * math.cos(lat_b) * math.cos(dlong))
            )

    if output_radians:
        return rad_bearing
    else:
        return math.degrees(rad_bearing)

def get_angle_between_lm_pts(l1, m1, l2, m2, output_radians=True):
    """
    Returns angle between points on the l,m plane (or any projection from a unit sphere, in which 
    case l,m are x,y)

    :Parameters:
    - l1: value of first coordinate in x direction
    - m1: value of first coordinate in y direction
    - l2: value of second coordinate in x direction
    - m2: value of second coordinate in y direction

    :Returns:
    - the angle in degrees or radius
    """

    # make sure everything is a numpy array
    l1 = np.array(l1)
    m1 = np.array(m1)
    l2 = np.array(l2)
    m2 = np.array(m2)

    n1 = np.sqrt(1-l1**2-m1**2)
    n2 = np.sqrt(1-l2**2-m2**2)

    dot = np.empty_like(l1)
    for i in np.arange(len(l1)):
        dot[i] = np.dot([l1[i],m1[i],n1[i]],[l2[i],m2[i],n2[i]])

    angle = np.arccos(dot)

    if output_radians:
        return angle
    else:
        return np.degrees(angle)

def lm_to_ea(l, m):
    '''
    Converts direction cosines to elevation and azimuth in radians.
    (new version)
    '''
    azimuth = np.arctan2(l, m)
    
    elev = np.arccos(np.sqrt(l**2 + m**2))

    return elev, azimuth