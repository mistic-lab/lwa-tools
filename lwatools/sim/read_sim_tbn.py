import numpy as np
import matplotlib.pyplot as plt

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile

from lwatools.visibilities.generate import compute_visibilities_gen
from lwatools.visibilities.baselines import uvw_from_antenna_pairs
from lwatools.imaging.imaging_utils import grid_visibilities, get_gimg_max
from lwatools.utils.array import select_antennas

tbn_fname = 'test.tbn'
station = stations.lwasv
use_pol = 0
tx_freq = 10e6 + 1000

tbnf = LWASVDataFile(tbn_fname, ignore_timetag_errors=True)

antennas = station.antennas
valid_ants, n_baselines = select_antennas(antennas, 0)

# overwrite the cable delays because we don't want compensation
for a in valid_ants:
    a.cable.delay = lambda *x: 0

vg = compute_visibilities_gen(tbnf, valid_ants, integration_length=1, fft_length=16)

bl, freqs, vis = next(vg)

uvw = uvw_from_antenna_pairs(bl, wavelength=3e8/tx_freq)

tbin = np.argmax([abs(f - tx_freq) for f in freqs])

vis_tbin = vis[:, tbin]

jd = tbnf.get_info('start_time').jd

gridded_image = grid_visibilities(bl, freqs, vis, tx_freq, station, jd=jd)

l, m, img, extent = get_gimg_max(gridded_image, return_img=True)

fig, ax = plt.subplots()
ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
plt.show()
