#!/usr/bin/python3

import argparse
import numpy as np
import time
import h5py

from mpi4py import MPI

from lsl.common import stations
from lsl.reader.ldp import LWASVDataFile
from lsl.correlator import fx as fxc
from lsl.correlator import uvutils
from lsl.reader.errors import EOFError

from lwatools.file_tools.outputs import build_output_file
from lwatools.file_tools.parseTBN import compute_number_of_integrations
from lwatools.utils.geometry import lm_to_ea
from lwatools.utils.array import select_antennas
from lwatools.utils import known_transmitters
from lwatools.visibilities.baselines import uvw_from_antenna_pairs
from lwatools.imaging.utils import get_gimg_max, get_gimg_center_of_mass, grid_visibilities

station = stations.lwasv

####
# TODO:
# - maybe pass a metadata dict to the worker before sending data
#   => frees up the MPI message tag for something other than integration number
#   => allows parameters (e.g. target bin) to change per-iteration
###

def main(args):
    # this first part of the code is run by all processes

    # set up MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise RuntimeError(f"This program requires at least two MPI processes to function. Please rerun with more resources")
    
    # designate the last process as the supervisor/file reader
    supervisor = size - 1

    # open the TBN file for reading
    tbnf = LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True)

    # figure out the details of the run we want to do
    tx_coords = known_transmitters.parse_args(args)
    antennas = station.antennas
    valid_ants, n_baselines = select_antennas(antennas, args.use_pol)
    n_ants = len(valid_ants)
    total_integrations = compute_number_of_integrations(tbnf, args.integration_length)

    sample_rate = tbnf.get_info('sample_rate')
    # some of our TBNs claim to have frame size 1024 but they are lying
    frame_size = 512
    tbn_center_freq = tbnf.get_info('freq1')


    # open the output HDF5 file and create datasets
    # because of the way parallelism in h5py works all processes (even ones
    # that don't write to the file) must do this
    h5f = build_output_file(args.hdf5_file, tbnf, valid_ants, n_baselines, args.integration_length, tx_freq=args.tx_freq, fft_len=args.fft_len, use_pfb=args.use_pfb, use_pol=args.use_pol, transmitter_coords=tx_coords, mpi_comm=comm)


    if args.point_finding_alg == 'all' or args.point_finding_alg == 'peak':
        h5f.create_dataset_like('l_peak', h5f['l_est'])
        h5f.create_dataset_like('m_peak', h5f['m_est'])
        h5f.create_dataset_like('elevation_peak', h5f['elevation'])
        h5f.create_dataset_like('azimuth_peak', h5f['azimuth'])
    if args.point_finding_alg == 'all' or args.point_finding_alg == 'CoM':
        h5f.create_dataset_like('l_CoM', h5f['l_est'])
        h5f.create_dataset_like('m_CoM', h5f['m_est'])
        h5f.create_dataset_like('elevation_CoM', h5f['elevation'])
        h5f.create_dataset_like('azimuth_CoM', h5f['azimuth'])
    else:
        raise NotImplementedError(f"Unrecognized point finding algorithm: {args.point_finding_alg}")
    del h5f['l_est']
    del h5f['m_est']
    del h5f['elevation']
    del h5f['azimuth']


    if rank == supervisor:
        # the supervisor process runs this code
        print("supervisor: started")

        # state info
        reached_end = False
        workers_alive = [True for _ in range(size - 1)]
        int_no = 0

        
        while True:
            if not reached_end:
                # grab data for the next available worker
                try:
                    duration, start_time, data = tbnf.read(args.integration_length)
                    # only use data from valid antennas
                    data = data[[a.digitizer - 1 for a in valid_ants], :]
                except EOFError:
                    reached_end = True
                    print(f"supervisor: reached EOF")
                if int_no >= total_integrations:
                    print(f"supervisor: this is the last integration")
                    reached_end = True

            # get the next "ready" message from the workers
            st = MPI.Status()
            msg = comm.recv(status=st)
            if msg == "ready":
                print(f"supervisor: received 'ready' message from worker {st.source}")
                
                # if we're done, send an exit message and mark that we've killed this worker
                # an empty array indicates that the worker should exit
                if reached_end:
                    print(f"supervisor: sending exit message to worker {st.source}")
                    comm.Send(np.array([]), dest=st.source, tag=int_no)
                    workers_alive[st.source] = False

                    if not any(workers_alive):
                        print(f"supervisor: all workers told to exit, goodbye")
                        break
                # otherwise, send the data to the worker for processing
                else:
                    print(f"supervisor: sending data for integration {int_no} to worker {st.source}")
                    # Send with a capital S is optimized to send numpy arrays
                    comm.Send(data, dest=st.source, tag=int_no)
                    int_no += 1
            else:
                raise ValueError(f"Supervisor received unrecognized message '{msg}' from worker {st.source}")

        tbnf.close()

    else:
        # the worker processes run this code
        print(f"worker {rank} started")

        # workers don't need access to the TBN file
        tbnf.close()
        
        # figure out the size of the incoming data buffer
        samples_per_integration = int(round(args.integration_length * sample_rate / frame_size)) * frame_size
        buffer_shape = (n_ants, samples_per_integration)

        while True:
            # send with a lowercase s can send any pickle-able python object
            # this is a synchronous send - it will block until the message is read by the supervisor
            # the other sends (e.g. comm.Send) only block until the message is safely taken by MPI, which might happen before the receiver actually reads it
            comm.ssend("ready", dest=supervisor)

            # build a buffer to be filled with data
            data = np.empty(buffer_shape, np.complex64)

            # receive the data from the supervisor
            st = MPI.Status()
            comm.Recv(data, source=supervisor, status=st)

            int_no = st.tag

            # if the buffer is empty, we're done
            if st.count == 0:
                print(f"worker {rank}: received exit message, exiting")
                break

            # otherwise process the data we've recieved
            print(f"worker {rank}: received data for integration {int_no}, starting processing")

            # run the correlator
            bl, freqs, vis = fxc.FXMaster(data, valid_ants, LFFT=args.fft_len, pfb=args.use_pfb, sample_rate=sample_rate, central_freq=tbn_center_freq, Pol='xx' if args.use_pol == 0 else 'yy', return_baselines=True, gain_correct=True)

            
            gridded_image = grid_visibilities(bl, freqs, vis, args.tx_freq, station)

            save_all_sky = (args.all_sky and int_no in args.all_sky) or (args.all_sky_every and int_no % args.all_sky_every == 0)


            if args.point_finding_alg == 'all' or 'peak':
                result = get_gimg_max(gridded_image, return_img=save_all_sky)
                l = result[0]
                m = result[1]
                src_elev, src_az = lm_to_ea(l, m)
                h5f['l_peak'][int_no] = l
                h5f['m_peak'][int_no] = m
                h5f['elevation_peak'][int_no] = src_elev
                h5f['azimuth_peak'][int_no] = src_az

            if args.point_finding_alg == 'all' or args.point_finding_alg == 'CoM':
                result = get_gimg_center_of_mass(gridded_image, return_img=save_all_sky)
                l = result[0]
                m = result[1]
                src_elev, src_az = lm_to_ea(l, m)
                h5f['l_CoM'][int_no] = l
                h5f['m_CoM'][int_no] = m
                h5f['elevation_CoM'][int_no] = src_elev
                h5f['azimuth_CoM'][int_no] = src_az

            if save_all_sky:
                img = result[2]
                extent = result[3]
                ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')
                plt.savefig('allsky_int_{}.png'.format(int_no))


            print(f"worker {rank}: done processing integration {int_no}")

            

    # back to common code for both supervisor and workers
    h5f.attrs['total_integrations'] = int_no
    h5f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute all-sky images and fit a model to them",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            fromfile_prefix_chars='@',
            epilog='''
This is a version of the imaging DoA estimation script
that uses MPI to run in parallel on a computer cluster.

To run this script on Compute Canada's slurm-based systems, create
an sbatch script. Select the number of MPI processes you want to
create using --ntasks=<number of processes>. Set the number of CPUs
per process using --cpus-per-task=<cpus/process>. In the sbatch
script, execute this python script using `srun python -m
lwatools.imaging.parallel_tbn_doa <args>`.

Alternatively, if you're working with a shell on the distributed
system already, such as if you've salloc'd some resources, you can
use `mpirun -n <number of processes> python -m
lwatools.imaging.parallel_tbn_doa <args>` to run this script.

This script's dependencies can be obtained on a Compute Canada
system by loading the following modules:
    - openmpi
    - mpi4py
    - hdf5-mpi

For more information on running MPI jobs on Compute Canada's clusters, see
https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling'''
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--hdf5-file', '-f', type=str, default='output.h5',
            help='name of output HDF5 file')
    parser.add_argument('--fft-len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use-pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use-pol', type=int, default=0,
            help='0 for X which is the only supported polarization')
    parser.add_argument('--integration-length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--all-sky', type=int, nargs='*',
            help='export all-sky plots for these integrations')
    parser.add_argument('--all-sky-every', type=int,
            help='export an all-sky plot every x integrations')
    parser.add_argument('--stop-after', type=int, default=-1,
            help='stop running after this many integrations')
    parser.add_argument('--point-finding-alg', nargs='?', default='all', choices=('peak', 'CoM', 'all'),
            help='select which algorithm is used to locate the point source in an image - options are the image peak or centre of mass')
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
