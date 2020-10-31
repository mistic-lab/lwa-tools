#!/usr/bin/python3

import argparse
import numpy as np
import h5py
from datetime import datetime
from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile
from lsl.imaging.data import VisibilityDataSet, PolarizationDataSet
from lsl.imaging.analysis import find_point_sources
from lsl.imaging.utils import build_gridded_image, plot_gridded_image
from lsl.sim import vis as simVis
from lsl.correlator.fx import pol_to_pols
from lsl.correlator import uvutils
import pickle

import matplotlib.pyplot as plt

from visibility_models import point_source_visibility_model_uv
from imaging_utils import lm_to_ea
from generate_visibilities import compute_visibilities_gen, select_antennas
import known_transmitters

station=stations.lwasv



def main(args):
    transmitter_coords = known_transmitters.parse_args(args)
    if transmitter_coords:
        bearing, _, distance = station.get_pointing_and_distance(transmitter_coords + [0])
    else:
        print("Please specify a transmitter location")
        return
    
    antennas = station.antennas

    valid_ants, n_baselines = select_antennas(antennas, args.use_pol)

    ## Build freqs
    freqs = np.empty((args.fft_len,),dtype=np.float64)
    #! Need to think of intelligent way of doing this.
    #! target_bin will probably not matter since all vis is the same
    freqs5 =   [5284999.9897182, 5291249.9897182, 5297499.9897182, 5303749.9897182,
                5309999.9897182, 5316249.9897182, 5322499.9897182, 5328749.9897182,
                5334999.9897182, 5341249.9897182, 5347499.9897182, 5353749.9897182,
                5359999.9897182, 5366249.9897182, 5372499.9897182, 5378749.9897182]
    for i in range(len(freqs)):
        freqs[i]=freqs5[i]

    ## Build bl
    pol_string = 'xx' if args.use_pol == 0 else 'yy'
    pol1, pol2 = pol_to_pols(pol_string)
    antennas1 = [a for a in valid_ants if a.pol == pol1]
    antennas2 = [a for a in valid_ants if a.pol == pol2]

    nStands = len(antennas1)
    baselines = uvutils.get_baselines(antennas1, antennas2=antennas2, include_auto=False, indicies=True)

    antennaBaselines = []
    for bl in range(len(baselines)):
            antennaBaselines.append( (antennas1[baselines[bl][0]], antennas2[baselines[bl][1]]) )
    bl = antennaBaselines

    uvw_m = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y, b[0].stand.z - b[1].stand.z]) for b in bl])
    uvw = np.empty((len(bl), 3, len(freqs)))
    for i, f in enumerate(freqs):
        # wavelength = 3e8/f # TODO this should be fixed. What is currently happening is not true. Well it is, but only if you're looking for a specific transmitter frequency. Which I guess we are. I just mean it's not generalized.
        wavelength = 3e8/args.tx_freq
        uvw[:,:,i] = uvw_m/wavelength

    ## Build vis
    vismodel = point_source_visibility_model_uv(uvw[:,0,0],uvw[:,1,0],args.l_model,args.m_model)
    vis = np.empty((len(vismodel), len(freqs)), dtype=np.complex64)
    for i in np.arange(vis.shape[1]):
        vis[:,i] = vismodel

    if args.export_npy:
        print("Exporting modelled u, v, w, and visibility")
        np.save('model-uvw.npy', uvw)
        np.save('model-vis.npy', vis)

    ## Start to build up the data structure for VisibilityDataSet
    # we only want the bin nearest to our frequency
    target_bin = np.argmin([abs(args.tx_freq - f) for f in freqs])

    # This can't matter, right?
    # jd = tbnf.get_info('start_time').jd
    jd = 2458847.2362531545

    # Build antenna array
    antenna_array = simVis.build_sim_array(station, antennas, freqs/1e9, jd=jd, force_flat=True)

    dataSet = VisibilityDataSet(jd=jd, freq=freqs, baselines=bl, uvw=uvw, antennarray=antenna_array)
    if args.use_pol == 0:
        pol_string = 'XX'
        p=0 # this is related to the enumerate in lsl.imaging.utils.CorrelatedIDI().get_data_set() (for when there are multiple pols in a single dataset)
    else:
        raise RuntimeError("Only pol. XX supported right now.")
    polDataSet = PolarizationDataSet(pol_string, data=vis)
    dataSet.append(polDataSet)

    sizes = [int(item) for item in args.size.split(',')]
    reses = [float(item) for item in args.res.split(',')]
    wreses = [float(item) for item in args.wres.split(',')]
    if len(sizes) != len(reses) and len(sizes) != len(wreses):
        raise RuntimeError("size, res and wres must all have same number of inputs")
    all_grid_params=[]
    for i in range(len(sizes)):
        all_grid_params.append({'size':sizes[i], 'res':reses[i], 'wres':wreses[i]})

    if args.all_sky:
        fig, ax = plt.subplots()

    # Iterate over size/res/wres and generate multiple grids/images   
    for grid_params in all_grid_params:
        print('| Gridding and imaging with size={}, res={}, wres={}'.format(
                grid_params['size'],grid_params['res'], grid_params['wres']))

        gridded_image = build_gridded_image(dataSet, pol=pol_string,
            chan=target_bin, size=grid_params['size'],
            res=grid_params['res'], wres=grid_params['wres'])
        
        if args.export_npy:
            print("Exporting gridded u, v, and visibility")
            u,v = gridded_image.get_uv()
            np.save('gridded-u-size-{}-res-{}-wres-{}.npy'.format(grid_params['size'],grid_params['res'], grid_params['wres']), u)
            np.save('gridded-v-size-{}-res-{}-wres-{}.npy'.format(grid_params['size'],grid_params['res'], grid_params['wres']), v)
            np.save('gridded-vis-size-{}-res-{}-wres-{}.npy'.format(grid_params['size'],grid_params['res'], grid_params['wres']), gridded_image.uv)

        # Plot/extract l/m do some modelling
        # I've largely borrow this from plot_gridded_image
        img = gridded_image.image()
        imgSize = img.shape[0]
        img = np.roll(img, imgSize//2, axis=0)
        img = np.roll(img, imgSize//2, axis=1)
        l, m = gridded_image.get_LM()
        extent = (m.max(), m.min(), l.min(), l.max())
        l = np.linspace(l.min(), l.max(), img.shape[0])
        m = np.linspace(m.min(), m.max(), img.shape[1])
        if l.shape != m.shape:
            raise RuntimeError("gridded_image is not a square")

        row, col = np.where(img == img.max())
        if len(row) == 1 and len(col) == 1:
            row = row[0]
            col = col[0]
        else:
            raise RuntimeError("Nick messed up. There are two maxes. This method won't work.")

        #! Note the negative
        l = l[-col]
        m = m[row]

        if args.all_sky:
            ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
            ax.set_title('size={}, res={}, wres={}'.format(
                grid_params['size'],grid_params['res'], grid_params['wres']))
            ax.set_xlabel('l')
            ax.set_ylabel('m')
            ax.plot(l,m,marker='o', color='k', label='Image Max.')
            ax.plot(args.l_model,args.m_model,marker='x', color='r', label='Model (input)')
            plt.legend(loc='lower right')
            plt.savefig('allsky_size_{}_res_{}_wres_{}.png'.format(
                grid_params['size'],grid_params['res'], grid_params['wres']))
            plt.cla()


        save_pkl_gridded = args.pkl_gridded and k in args.pkl_gridded
        if save_pkl_gridded:
            quickDict={'image':img, 'extent':extent}
            with open('gridded_size_{}_res_{}_wres_{}.pkl'.format(
                grid_params['size'],grid_params['res'], grid_params['wres']),
                'wb') as f:
                pickle.dump(quickDict, f, protocol=pickle.HIGHEST_PROTOCOL)



    # h5f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute all-sky images and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('--l-model','-l', type=float, default=0.22,
            help='l to be used for model')
    parser.add_argument('--m-model','-m', type=float, default=0.32,
            help='m to be used for model')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--fft_len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use_pol', type=int, default=0,
            help='Jeff what is this')
    parser.add_argument('--integration_length', type=float, default=1,
            help='Integration length in seconds')
    # parser.add_argument('-l', '--list', help='delimited list input', type=str)
    parser.add_argument('--size', type=str, default='100',
            help='Sizes of UV matrices in wavelengths')
    parser.add_argument('--res', type=str, default='1',
            help='Resolution of UV matrices')
    parser.add_argument('--wres',type=str, default='0.5',
            help='Gridding resolution of sqrt(w) when projecting to w=0')

    # parser.add_argument('--size', type=int, nargs='*', default=100,
    #         help='Sizes of UV matrices in wavelengths')
    # parser.add_argument('--res', type=float, nargs='*', default=1,
    #         help='Resolution of UV matrices')
    # parser.add_argument('--wres',type=float, nargs='*', default=0.5,
    #         help='Gridding resolution of sqrt(w) when projecting to w=0')
    parser.add_argument('--all-sky', type=bool, default=False,
            help='export all-sky plots')
    parser.add_argument('--pkl-gridded', type=int, nargs='*',
            help='export gridded all sky data for these integrations')
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
