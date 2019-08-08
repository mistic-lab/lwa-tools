import numpy as np

def make_complex_sine(f, fs, t, phi):
    """Makes complex sine wave.

    Parameters
    ----------
    f : float
                frequency of sine
    fs : float
                sampling rate
    t : float
                length of signal in seconds
    phi : float
                phase offset in radians

    Returns
    -------
    numpy array
        sampled sine wave
    """
    t_arr = np.linspace(0,t,t*fs) # time in s
    omega = f * 2 * np.pi
    phi = 0 # rads
    x = np.exp(1j*(omega * t_arr + phi))

    return x
