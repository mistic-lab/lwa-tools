# Some tools to help build SDF files. The SDF GUI tool doesn't seem to work for the lower frequencies for LWA-SV.
# A description of the various parameters is at the following
# https://www.faculty.ece.vt.edu/swe/lwavt/doc/MCS0030v3_Observing.pdf

def get_OBS_FREQ(mhz:float):
    """
    For finding OBS_FREQ and OBS_FREQ+. Both are printed with the correct number of digits.

    Parameters:
    -----------

    mhz: float
        MHZ to find the goodies from. Can be with any number of digits. Give the optimal (ie 5.025)
    """
    DP_range = round( mhz * (2**32) / 196 )
    print("DP_RANGE: {}".format(DP_range))
    freq = DP_range * 196 / (2**32)
    print("FREQ: {:.9f}".format(freq))

def get_OBS_START_MPM(hour:int, minute:int, second:int):
    """
    For finding OBS_START_MPM. Printed with the correct number of digits.

    Parameters:
    -----------

    hour: int
        hours since midnight (use 24 hour time)
    minute: int
        minutes on the hour
    second: int
        seconds on the minute
    """
    hour_ms = hour * 60 * 60 * 1000
    minute_ms = minute * 60 * 1000
    second_ms = second * 1000
    MPM = hour_ms + minute_ms + second_ms
    print("MPM: {}".format(MPM))