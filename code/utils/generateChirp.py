import numpy as np
from scipy.signal import chirp


def generateChirp(fs, f0, f1, T, dtype='complex'):
    """
    Outputs a lfm (linear frequency moduluated) signal. Geeks call them 'chirp'
    signals.

    Parameters
    ----------
    fs : int or flt
        Sampling rate of the signal [Hz].
    f0 : int or float
        Starting frequency of the sweep [Hz].
    f1 : int or float
        Ending frequency of the sweep [Hz].
    T : int or float
        Length of the sweep [s].
    dtype : str (default: 'complex')
        Datatype of the signal. Options are 'complex' and 'float'.

    """
    print('GENERATECHIRP:')

    t = np.linspace(0, T, fs*T)
    if (dtype == 'float'):
        sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')
    elif (dtype == 'complex'):
        sweep = np.exp(1j*(np.pi*((f1-f0)/T)*np.square(t)))

    print('--> sweep.dtype: {}'.format(sweep.dtype))

    return sweep
