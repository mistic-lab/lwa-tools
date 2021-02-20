#!/usr/bin/python3

import argparse
import sys
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

from lwatools.vis_modeling.visibility_models import point_source_visibility_model_uv
from lwatools.imaging.imaging_utils import lm_to_ea, get_gimg_max
from lwatoosl.ionospheric_models.fixed_dist_mirrors import flatmirror_height
from lwatools.vis_modeling.generate_visibilities import compute_visibilities_gen, select_antennas
from lwatools.utils import known_transmitters

station=stations.lwasv

def main(args):

    ## Check we should bother doing anything

    if not args.export_npy and not args.export_h5 and not args.all_sky and not args.pkl_gridded:
        raise RuntimeError("You have not selected a data output of any type. Read the docstring and pick something for me to do.")

    # Normalize all inputs to the same length
    sizes  = [int(item) for item in args.size.split(',')]
    reses  = [float(item) for item in args.res.split(',')]
    wreses = [float(item) for item in args.wres.split(',')]
    maxinputlen = max(len(sizes), len(reses), len(wreses))
    if len(sizes) not in [1, maxinputlen] or len(reses) not in [1, maxinputlen] or len(wreses) not in [1,maxinputlen]:
        raise RuntimeError(" \
        For size, res and wres you must pass either the same number of values as the max or a single value.\n \
        For example:\n \
        ALLOWED     -> sizes=175,180,190, res=0.5, wres=0.5\n \
                    -> sizes=175,180,190, res=0.5, wres=0.5,0.6,0.7\n \
        NOT ALLOWED -> sizes=175,180,190, res=0.5, wres=0.5,0.6 \
        ")
    if len(sizes) != maxinputlen: # You'd think there must be a good way to do this with list comprehension.
        sizes = sizes * maxinputlen
    if len(reses) != maxinputlen:
        reses = reses * maxinputlen
    if len(wreses) != maxinputlen:
        wreses = wreses * maxinputlen
    all_grid_params=[]
    while len(sizes) > 0:
        all_grid_params.append({'size':sizes.pop(), 'res':reses.pop(), 'wres':wreses.pop()})


    ## Begin doing stuff
    transmitter_coords = known_transmitters.parse_args(args)
    if transmitter_coords:
        bearing, _, distance = station.get_pointing_and_distance(transmitter_coords + [0])
    else:
        print("Please specify a transmitter location")
        return
    
    antennas = station.antennas

    valid_ants, n_baselines = select_antennas(antennas, args.use_pol)


    if args.export_h5:
        h5fname="simulation-results.h5"
        print("Output will be written to {}".format(h5fname))
        h5f=h5py.File(h5fname,'w')

        ats = h5f.attrs
        ats['transmitter'] = args.transmitter
        ats['tx_freq'] = args.tx_freq
        ats['valid_ants'] = [a.id for a in valid_ants]
        ats['n_baselines'] = n_baselines
        ats['fft_len'] = args.fft_len
        ats['use_pol'] = args.use_pol
        ats['int_length'] = args.integration_length
        ats['l_model'] = args.l_model
        ats['m_model'] = args.m_model

        h5f.create_dataset('l_est', (len(all_grid_params),))
        h5f.create_dataset('m_est', (len(all_grid_params),))
        h5f.create_dataset('wres', (len(all_grid_params),))
        h5f.create_dataset('res', (len(all_grid_params),))
        h5f.create_dataset('size', (len(all_grid_params),))
        h5f.create_dataset('extent', (len(all_grid_params),4))
        h5f.create_dataset('elevation', (len(all_grid_params),))
        h5f.create_dataset('azimuth', (len(all_grid_params),))
        h5f.create_dataset('height', (len(all_grid_params),))





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
        print(args.export_npy)
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

    if args.all_sky:
        fig, ax = plt.subplots()

    # Iterate over size/res/wres and generate multiple grids/images   
    k=0
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


        l,m,img,extent=get_gimg_max(gridded_image, return_img=True)

        # Compute other values of interest
        elev, az = lm_to_ea(l, m)
        height = flatmirror_height(elev, distance)

        if args.export_h5:
            h5f['l_est'][k] = l
            h5f['m_est'][k] = m
            h5f['wres'][k] = grid_params['wres']
            h5f['res'][k] = grid_params['res']
            h5f['size'][k] = grid_params['size']

            h5f['extent'][k] = extent

            h5f['elevation'][k] = elev
            h5f['azimuth'][k] = az
            h5f['height'][k] = height

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
        k+=1


    if args.export_h5:
        h5f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="simulate data then image with varying sizes/reses/wreses",
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
    parser.add_argument('--size', type=str, default='100',
            help='Sizes of UV matrices in wavelengths')
    parser.add_argument('--res', type=str, default='1',
            help='Resolution of UV matrices')
    parser.add_argument('--wres',type=str, default='0.5',
            help='Gridding resolution of sqrt(w) when projecting to w=0')
    parser.add_argument('--all-sky', action='store_true',
            help='export all-sky plots')
    parser.add_argument('--pkl-gridded', type=int, nargs='*',
            help='export gridded all sky data for these integrations')
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration")
    parser.add_argument('--export-h5', action='store_true',
            help="export result to an h5 file")
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
