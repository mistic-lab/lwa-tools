import numpy as np
import h5py
import matplotlib.pyplot as plt
import math

def running_mean(x, N): 
    """ 
    Returns array. Calculates rolling average of length N on array x. 
    """ 
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def find_virtual_height_plane_wave(phase_diff_m, x1, x2):
    """
    Return height of ionosphere

    Parameters
    ----------
    phase_diff_m : float
        difference in receiving path length in m
    x1 : float
        distance from transmitter to first antenna
    x2 : float
        distance from first antenna to second antenna
    """

    # phase_diff_m = phase_diff_m/2  # Half wave limit (fx correlator causes pi to mean pi/2)

    # acosarg = phase_diff_m/x2
    # # acosarg=acosarg.astype(float)
    # acosarg[acosarg < -1.0] = np.nan
    # acosarg[acosarg > 1.0] = np.nan
    # acosarg = np.maximum(acosarg, -1)
    # acosarg = np.minimum(acosarg, 1)

    x3 = x1+x2  # Distance from TX to second antenna

    theta = np.arccos(phase_diff_m/x2)
    # print("angle of arrival: {}".format(math.degrees(theta)))
    low = np.tan(theta)*x1/2
    high = np.tan(theta)*x3/2
    avg = np.empty_like(low)
    for i in range(len(avg)):
        avg[i] = abs(high[i]-low[i])/2 + min(high[i], low[i])
    avg[avg>20e5] = np.nan
    avg[avg<-20e5] = np.nan
    return avg

def hacky_VH(phase_diff_m, x1, x2):
    """
    x1 is tx->224
    x2 is 224->ant
    """
    l = x1/x2 * phase_diff_m/2
    sqrt_arg = - l**2 + (x1**2)/4
    # sqrt_arg[sqrt_arg<0]=0
    # print(np.count_nonzero(sqrt_arg))
    h = np.sqrt(sqrt_arg)
    return h

def get_phase_diff_as_dist(phase_diff_rad, frequency):
    """
    Return distance in m

    Parameters
    ----------
    phase_diff_rad : float
        difference in receiving path length in rad
    frequency : int
        frequency to calculate it at in Hz (optional)
    """
    c = 3e8
    wavelength = c/(frequency)
    phase_diff_m = wavelength*phase_diff_rad/(2*math.pi)
    return phase_diff_m


f = h5py.File('058846_000123426_ACM.hdf5','r')
pol0 = f['pol0']
a141=np.angle(pol0[3,0,:])
a178=np.angle(pol0[3,1,:])
a180=np.angle(pol0[3,2,:])

#all in m
d180=20.476
d178=27.973
d141=41.747
d224_tx = 171300

fc_desired = 5351500 # the actual signal
fs = f.attrs['sampleRate']
df = fs/f.attrs['nFFT']
fc_bin=f.attrs['freq1']-fs/2+f.attrs['fBin']*df
print('Bin center to signal center offset: {}'.format(fc_bin))


# Get a running mean to smooth things out
a141 = running_mean(a141, 300)
a178 = running_mean(a178, 300)
a180 = running_mean(a180, 300)

fig, ax = plt.subplots()
ax.plot(a141, label='141')
ax.plot(a178, label='178')
ax.plot(a180, label='180')
ax.set_title('Running mean (300) of 2.5 Hz relative phases (ref 224)')
ax.set_xlabel('Samples')
ax.set_ylabel('Phase difference (rads)')
plt.legend()
plt.show(block=False)



a141_m = get_phase_diff_as_dist(a141, fc_desired)
a178_m = get_phase_diff_as_dist(a178, fc_desired)
a180_m = get_phase_diff_as_dist(a180, fc_desired)

fig, ax = plt.subplots()
ax.plot(a141_m, label='141')
ax.plot(a178_m, label='178')
ax.plot(a180_m, label='180')
ax.set_title('Phase diff in m. (ref 224)')
ax.set_xlabel('Samples')
ax.set_ylabel('Phase difference (m)')
plt.legend()
plt.show(block=False)

a141_h = find_virtual_height_plane_wave(a141_m, d224_tx, d141)
a178_h = find_virtual_height_plane_wave(a178_m, d224_tx, d178)
a180_h = find_virtual_height_plane_wave(a180_m, d224_tx, d180)
# a141_h = hacky_VH(a141_m, d224_tx, d141)
# a178_h = hacky_VH(a178_m, d224_tx, d178)
# a180_h = hacky_VH(a180_m, d224_tx, d180)

fig, ax = plt.subplots()
ax.plot(a141_h/1000, label='a141')
ax.plot(a178_h/1000, label='a178')
ax.plot(a180_h/1000, label='a180')
ax.set_title('Virtual height')
ax.set_xlabel('km')
ax.set_ylabel('Samples')
plt.legend()
plt.show()

f.close()