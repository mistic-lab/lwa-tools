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
from lwatools.utils.geometry import lm_to_ea
from lwatools.utils.array import select_antennas
from lwatools.utils import known_transmitters
from lwatools.visibilities.generate import compute_visibilities_gen
from lwatools.visibilities.baselines import uvw_from_antenna_pairs
from lwatools.visibilities.models import point_residual_abs, bind_gaussian_residual
from lwatools.visibilities.model_fitting import fit_model_to_vis
from lwatools.plot.vis import vis_phase_scatter_3d

opt_method = 'lm'

station = stations.lwasv

residual_function = bind_gaussian_residual(1)

# TODO: should probably read this from the TBN file, but the workers need it and they don't open the TBN
sample_rate = 100e3
frame_size = 512

####
# TODO:
# maybe pass a metadata dict to the worker before sending data
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

    # figure out the details of the run we want to do
    antennas = station.antennas
    valid_ants, n_baselines = select_antennas(antennas, args.use_pol)
    n_ants = len(valid_ants)


    # open the output HDF5 file and create datasets
    # because of the way parallelism in h5py works all processes (even ones
    # that don't write to the file) must do this
    h5f = h5py.File('/scratch/jtst/output.h5', 'w', driver='mpio', comm=comm)
    dset_rank = h5f.create_dataset('rank', (21,), dtype='i')
    dset_delay = h5f.create_dataset('delay', (21,), dtype='f')


    if rank == supervisor:
        # the supervisor process runs this code
        print("supervisor: started")

        # state info
        reached_end = False
        workers_alive = [True for _ in range(size - 1)]
        int_no = 0

        # open the TBN file for reading
        tbnf = LWASVDataFile(args.tbn_filename, ignore_timetag_errors=True)
        
        while True:
            if not reached_end:
                # grab data for the next available worker
                try:
                    duration, start_time, data = tbnf.read(args.integration_length)
                except EOFError:
                    reached_end = True
                    print(f"supervisor: reached EOF")

            # get the next "ready" message from the workers
            st = MPI.Status()
            msg = comm.recv(status=st)
            if msg == "ready":
                print(f"supervisor: received 'ready' message from worker {st.source}")
                
                # if we're done, send an exit message and mark that we've killed this worker
                # an empty array indicates that the worker should exit
                if reached_end:
                    print(f"supervisor: sending exit message to worker {st.source}")
                    comm.Send(np.array([]), dest=st.source)
                    workers_alive[st.source] = False

                    if not any(workers_alive):
                        print(f"supervisor: all workers told to exit, goodbye")
                        break
                # otherwise, send the data to the worker for processing
                else:
                    print(f"supervisor: sending data {data.shape} to worker {st.source}")
                    # Send with a capital S is optimized to send numpy arrays
                    comm.Send(data, dest=st.source, tag=int_no)
                    int_no += 1
            else:
                raise ValueError(f"Supervisor received unrecognized message '{msg}' from worker {st.source}")

        tbnf.close()

    else:
        # the worker processes run this code
        print(f"worker {rank} started")
        
        # figure out the size of the incoming data buffer
        samples_per_integration = int(args.integration_length * sample_rate / frame_size) * frame_size
        buffer_shape = (512, samples_per_integration)

        while True:
            # send with a lowercase s can send any pickle-able python object
            # this is a synchronous send - it will block until the message is read by the supervisor
            # the other sends (e.g. comm.Send) only block until the message is safely taken by MPI, which might happen before the receiver actually reads it
            comm.ssend("ready", dest=supervisor)

            # build a buffer to be filled with data
            d = np.empty(buffer_shape, np.complex64)

            # receive the data from the supervisor
            st = MPI.Status()
            comm.Recv(d, source=supervisor, status=st)

            int_no = st.tag

            # if the buffer is empty, we're done
            if st.count == 0:
                print(f"worker {rank}: received exit message, exiting")
                break

            # otherwise process the data we've recieved
            print(f"worker {rank}: received data {d.shape} from supervisor, starting processing")
            delay = np.random.rand() * 5
            print(f"worker {rank}: processing for {delay} seconds")
            time.sleep(delay)

            dset_rank[int_no] = rank
            dset_delay[int_no] = delay

            print(f"worker {rank}: done processing")

    # back to common code for both supervisor and workers

    h5f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="compute visibilities and fit a model to them",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('tbn_filename', type=str,
            help='name of TBN data file')
    parser.add_argument('tx_freq', type=float,
            help='transmitter frequency')
    parser.add_argument('--hdf5-file', '-f', type=str, default='output.h5',
            help='name of output HDF5 file')
    parser.add_argument('--l-guess', type=float, default=0.0,
            help='initial guess for l parameter')
    parser.add_argument('--m-guess', type=float, default=0.0,
            help='initial guess for m parameter')
    parser.add_argument('--fft-len', type=int, default=16,
            help='Size of FFT used in correlator')
    parser.add_argument('--use-pfb', action='store_true',
            help='Whether to use PFB in correlator')
    parser.add_argument('--use-pol', type=int, default=0,
            help='0 for X which is the only supported polarization')
    parser.add_argument('--integration-length', type=float, default=1,
            help='Integration length in seconds')
    parser.add_argument('--scatter', type=int, nargs='*',
            help='export scatter plots for these integrations - warning: each scatter plot is about 6MB')
    parser.add_argument('--scatter-every', type=int,
            help='export a scatter plot every x integrations')
    parser.add_argument('--exclude', type=int, nargs='*',
            help="don't use these integrations in parameter guessing")
    parser.add_argument('--export-npy', action='store_true',
            help="export npy files of u, v, and visibility for each iteration - NOTE: these will take up LOTS OF SPACE if you run an entire file with this on!")
    #parser.add_argument('--visibility-model', default='gaussian',
    #        choices=('point', 'gaussian', 'chained'),
    #        help='select what kind of model is fit to the visibility data')
            
    known_transmitters.add_args(parser)
    args = parser.parse_args()
    main(args)
