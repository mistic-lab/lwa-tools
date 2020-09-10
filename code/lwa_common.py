#!/usr/bin/python3

import math

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
