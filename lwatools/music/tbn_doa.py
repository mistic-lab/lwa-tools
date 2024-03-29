#!/usr/bin/python3
'''
Uses the MUSIC algorithm to estimate direction of arrival from LWA TBN files.
'''

import argparse
import aipy
import numpy as np
import h5py

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile
from astropy.constants import c #as speedOfLight
speedOfLight = c.value

from lwatools.file_tools.outputs import build_output_file
from lwatools.utils.array import select_antennas
from lwatools.utils import known_transmitters
from lwatools.utils.geometry import lm_to_ea


def main(args):

        # saz and sel are used later 
        img = aipy.img.ImgW(size=50, res=0.5)
        top = img.get_top(center=(50,50))
        saz,sel = aipy.coord.top2azalt(top)

        station=stations.lwasv

        tx_coords = known_transmitters.parse_args(args)

        print("Opening TBN file ({})".format(args.tbn_filename))
        with LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True) as tbnf:

                antennas = station.antennas

                valid_ants, n_baselines = select_antennas(antennas, args.use_pol)

                if not args.hdf5_file:
                        raise RuntimeError('Please provide an output filename')
                else:
                        with build_output_file(h5_fname=args.hdf5_file, tbnf=tbnf,
                                        valid_ants=valid_ants, n_baselines=n_baselines,
                                        # use_pfb=args.use_pfb, use_pol=args.use_pol, 
                                        integration_length=args.integration_length, transmitter_coords=tx_coords) as h5f:

                                del h5f['l_est']
                                del h5f['m_est']

                                freq = tbnf.get_info('freq1')
                                idx = [ant.digitizer-1 for ant in valid_ants]
                                xyz = np.array([[ant.stand.x, ant.stand.y, ant.stand.z] for ant in valid_ants])
                                delays = np.array([ant.cable.delay(freq) for ant in valid_ants])
                                delays -= delays.min()

                                n_samples = tbnf.get_info('nframe') / tbnf.get_info('nantenna')
                                samples_per_integration = int(args.integration_length * tbnf.get_info('sample_rate') / 512)
                                n_integrations = int(np.floor(n_samples / samples_per_integration))


                                for int_num in range(n_integrations):
                                        print(f"Starting iteration {int_num + 1} of {n_integrations}")
                                        # Load in the data and select what we need
                                        tInt, t0, data = tbnf.read(args.integration_length)

                                        data = data[idx,:]

                                        # Apply a phase rotation to deal with the cable delays
                                        for i in range(data.shape[0]):
                                                data[i,:] *= np.exp(2j*np.pi*freq*delays[i])
                                        data /= (np.abs(data)).max()




                                        # Calculate Rx - The time-averaged autocorrelation matrix
                                        nSamp = data.shape[1]
                                        xOutput = []
                                        print("Computing time-averaged autocorrelation matrix")
                                        for i in range(nSamp):
                                                x = np.matrix( data[:,i] ).T
                                                xOutput.append( x )
                                                try:
                                                        Rx += x*x.H
                                                except:
                                                        Rx =  x*x.H
                                        Rx /= nSamp

                                        # Find the eigenvectors/values for Rx and order them by significance
                                        print("Computing eigenvectors/values of the ACM")
                                        w, v = np.linalg.eig(Rx)
                                        order = np.argsort(np.abs(w))[::-1]
                                        w = w[order]
                                        v = v[:,order]

                                        # Break the eigenvalues into a signal sub-space, Us, and a noise sub-
                                        # space, Un.  This is currently done based on the number of sources
                                        # we have rather than inferred from the eigenvalues.
                                        ##Us = numpy.where( numpy.abs(w) > sigma )[0] #TODO I think this part should help find frequency too but Jayce had it commented out because the sigma section wasn't working (see her tbnMusic.py script I think)
                                        ##Un = numpy.where( numpy.abs(w) <= sigma )[0]
                                        # Us = range(3) #TODO What Jayce had. I imagine she had 4 sources
                                        # Un = range(3, w.size)
                                        Us = range(1)
                                        Un = range(1, w.size)

                                        print("Evaluating MUSIC spectrum")
                                        P = np.zeros_like(saz)
                                        E = np.zeros_like(saz)
                                        for i in range(saz.shape[0]):
                                                print(f"Starting row {i+1} / {saz.shape[0]} for integration {int_num}")
                                                for j in range(saz.shape[1]):
                                                        ta = saz[i,j]
                                                        te = sel[i,j]
                                                        if not np.isfinite(ta) or not np.isfinite(te):
                                                                continue
                                                                
                                                        pv = np.array([np.cos(te)*np.sin(ta), 
                                                                                np.cos(te)*np.cos(ta),
                                                                                np.sin(te)])
                                                                                        
                                                        a = np.zeros((len(valid_ants),1), dtype=np.complex128)
                                                        for k in range(len(valid_ants)):
                                                                a[k,0] = np.exp( 2j*np.pi*freq*np.dot(xyz[k,:]-xyz[0,:], pv)/speedOfLight )
                                                        a = np.matrix(a)
                                                                                        
                                                        v2 = np.matrix(v[:,Un])
                                                        o = a.H*v2*v2.H*a
                                                        P[i,j] = 1.0/max([1e-9, o[0,0].real])
                                                        
                                        spectrum_max_idx = np.where(P == P.max())
                                        el_max = sel[spectrum_max_idx][0]
                                        az_max = saz[spectrum_max_idx][0]
                                        h5f['elevation'][int_num] = el_max
                                        h5f['azimuth'][int_num] = az_max
                                        print(f"Integration complete - az = {az_max:.2f} el = {el_max:.2f}")


if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                        description="compute MUSIC results",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        fromfile_prefix_chars='@'
                        )
        parser.add_argument('tbn_filename', type=str,
                        help='name of TBN data file')
        parser.add_argument('--hdf5-file', '-f', type=str,
                        help='name of output HDF5 file')
        # parser.add_argument('--fft-len', type=int, default=16,
        #               help='Size of FFT used in correlator')
        # parser.add_argument('--use-pfb', action='store_true',
        #         help='Whether to use PFB in correlator')
        parser.add_argument('--use-pol', type=int, default=0,
                        help='0 means X and is the only currently supported option')
        parser.add_argument('--integration-length', type=float, default=1.0,
                        help='Integration length in seconds')
        parser.add_argument('--stop-after', type=int, default=-1,
                        help='stop running after this many integrations')
                        
        known_transmitters.add_args(parser)
        args = parser.parse_args()
        main(args)
