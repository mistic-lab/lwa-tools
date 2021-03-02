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
from lwatools.ionospheric_models.fixed_dist_mirrors import flatmirror_height, tiltedmirror_height
from lwatools.vis_modeling.generate_visibilities import compute_visibilities_gen, select_antennas

station=stations.lwasv


def main(args):

    h5fi = h5py.File(args.input_file, 'r')
    h5fo = h5py.File(args.output_file,'w')

    # Copy over important attributes
    for key in h5fi.attrs.keys():
        h5fo.attrs[key]=h5fi.attrs[key]
    
    # Copy over important datasets
    for key in h5fi.keys():
        temp_arr = h5fi[key]
        h5fo.create_dataset('vis_{}'.format(key),data=temp_arr)
    h5fo.attrs['grid_size']=args.size
    h5fo.attrs['grid_res']=args.res
    h5fo.attrs['grid_wres']=args.wres
    h5fo.create_dataset('l_est', (len(h5fo['vis_l_est']),))
    h5fo.create_dataset('m_est', (len(h5fo['vis_l_est']),))
    h5fo.create_dataset('extent', (len(h5fo['vis_l_est']),4))
    h5fo.create_dataset('elevation', (len(h5fo['vis_l_est']),))
    h5fo.create_dataset('azimuth', (len(h5fo['vis_l_est']),))
    h5fo.create_dataset('height', (len(h5fo['vis_l_est']),))

    
    h5fi.close() # done with input data now

    ## Begin doing stuff
    antennas = station.antennas
    valid_ants, n_baselines = select_antennas(antennas, h5fo.attrs['use_pol'], exclude=[256]) # to exclude outrigger

    tx_coords = h5fo.attrs['tx_coordinates']
    rx_coords = [station.lat * 180/np.pi, station.lon * 180/np.pi]

    ## Build freqs (same for every 'integration')
    freqs = np.empty((h5fo.attrs['fft_len'],),dtype=np.float64)
    #! Need to think of intelligent way of doing this.
    #! target_bin will probably not matter since all vis is the same
    freqs5 =   [5284999.9897182, 5291249.9897182, 5297499.9897182, 5303749.9897182,
                5309999.9897182, 5316249.9897182, 5322499.9897182, 5328749.9897182,
                5334999.9897182, 5341249.9897182, 5347499.9897182, 5353749.9897182,
                5359999.9897182, 5366249.9897182, 5372499.9897182, 5378749.9897182]
    for i in range(len(freqs)):
        freqs[i]=freqs5[i]

    ## Build bl (same for every 'integration')
    pol_string = 'xx' if h5fo.attrs['use_pol'] == 0 else 'yy'
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
        wavelength = 3e8/h5fo.attrs['tx_freq']
        uvw[:,:,i] = uvw_m/wavelength


    # Build antenna array (gets used in the VisibilityDataSet)
    # jd can't matter, right?
    jd = 2458847.2362531545
    antenna_array = simVis.build_sim_array(station, valid_ants, freqs/1e9, jd=jd, force_flat=True)
    # we only want the bin nearest to our frequency
    target_bin = np.argmin([abs(h5fo.attrs['tx_freq'] - f) for f in freqs])


    # Needed for PolarizationDataSet
    if h5fo.attrs['use_pol'] == 0:
        pol_string = 'XX'
        p=0 # this is related to the enumerate in lsl.imaging.utils.CorrelatedIDI().get_data_set() (for when there are multiple pols in a single dataset)
    else:
        raise RuntimeError("Only pol. XX supported right now.")


    if args.all_sky:
        fig, ax = plt.subplots()

    for k in np.arange(len(h5fo['vis_l_est'])):
        l_in = h5fo['vis_l_est'][k]
        m_in = h5fo['vis_m_est'][k]

        ## Build vis
        vismodel = point_source_visibility_model_uv(uvw[:,0,0],uvw[:,1,0],l_in,m_in)
        vis = np.empty((len(vismodel), len(freqs)), dtype=np.complex64)
        for i in np.arange(vis.shape[1]):
            vis[:,i] = vismodel

        if args.export_npy:
            print(args.export_npy)
            print("Exporting modelled u, v, w, and visibility")
            np.save('model-uvw{}.npy'.format(k), uvw)
            np.save('model-vis{}.npy'.format(k), vis)

        ## Start to build up the data structure for VisibilityDataSet

        dataSet = VisibilityDataSet(jd=jd, freq=freqs, baselines=bl, uvw=uvw, antennarray=antenna_array)
        polDataSet = PolarizationDataSet(pol_string, data=vis)
        dataSet.append(polDataSet)


        print('| Gridding and imaging with size={}, res={}, wres={}'.format(args.size, args.res, args.wres))

        gridded_image = build_gridded_image(dataSet, pol=pol_string,
            chan=target_bin, size=args.size,
            res=args.res, wres=args.wres)
        
        if args.export_npy:
            print("Exporting gridded u, v, and visibility")
            u,v = gridded_image.get_uv()
            np.save('gridded-u{}.npy'.format(k), u)
            np.save('gridded-v{}.npy'.format(k), v)
            np.save('gridded-vis{}.npy'.format(k), gridded_image.uv)


        l,m,img,extent=get_gimg_max(gridded_image, return_img=True)

        # Compute other values of interest
        elev, az = lm_to_ea(l, m)
        height = flatmirror_height(tx_coords, rx_coords, elev)

        h5fo['l_est'][k] = l
        h5fo['m_est'][k] = m

        h5fo['extent'][k] = extent

        h5fo['elevation'][k] = elev
        h5fo['azimuth'][k] = az
        h5fo['height'][k] = height

        if args.all_sky:
            ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
            ax.set_title('size={}, res={}, wres={}, iteration={}'.format(args.size,args.res,args.wres,k))
            ax.set_xlabel('l')
            ax.set_ylabel('m')
            ax.plot(l,m,marker='o', color='k', label='Image Max.')
            ax.plot(l_in,m_in,marker='x', color='r', label='Model (input)')
            plt.legend(loc='lower right')
            plt.savefig('allsky{}.png'.format(k))
            plt.cla()

        save_pkl_gridded = args.pkl_gridded and k in args.pkl_gridded
        if save_pkl_gridded:
            quickDict={'image':img, 'extent':extent}
            with open('gridded{}.pkl'.format(k), 'wb') as f:
                pickle.dump(quickDict, f, protocol=pickle.HIGHEST_PROTOCOL)

    h5fo.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Take existing vis_model_fitting output h5, extract l and m and use those as simulated points for image_model_fitting",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('--input-file','-fi', type=str,
            help='input file from which to get l and m arrays')
    parser.add_argument('--output-file','-fo', type=str,
            help='output file of simulated and gridded data')
    parser.add_argument('--size', type=float, default=80,
            help='Size of UV matrix in wavelengths')
    parser.add_argument('--res', type=float, default=0.5,
            help='Resolution of UV matrix')
    parser.add_argument('--wres',type=float, default=0.1,
            help='Gridding resolution of sqrt(w) when projecting to w=0')
    parser.add_argument('--all-sky', action='store_true',
            help='export all-sky plots')
    parser.add_argument('--pkl-gridded', type=int, nargs='*',
            help='export gridded all sky data for these integrations')
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration")
    args = parser.parse_args()
    main(args)
