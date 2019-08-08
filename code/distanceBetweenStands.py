#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Example script to read in the positions of stands at LWA-1 and make a plot
of the site."""

#Edited by Nicholas Bruce to show angle of wavefront arrival from digisonde

import sys
import numpy as np
import argparse
import math

from lsl.common import stations, metabundle, metabundleADP

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import linearAlgebraTools as lat

def main(args):
    # Parse command line
    toMark = np.array(args.stand)-1

    # Setup the LWA station information
    if args.metadata is not None:
        try:
            station = stations.parseSSMIF(args.metadata)
        except ValueError:
            try:
                station = metabundle.getStation(args.metadata, ApplySDM=True)
            except:
                station = metabundleADP.getStation(args.metadata, ApplySDM=True)
    elif args.lwasv:
        station = stations.lwasv
    else:
        station = stations.lwa1
    stands = station.getStands()

    # data format 
    # ----| 0: stand id
    # ----| 1: x coord
    # ----| 2: y coord
    # ----| 3: z coord
    # ----| 4: dist to line
    # ----| 5: x coord closest on line
    # ----| 6: y coord closest on line
    data = np.zeros((len(args.stand),7))
    print "--| Number of stands to plot: %i" % len(data)

    wavefront_max_Y = 0
    wavefront_min_Y = 0
    stands_max_Z = 0
    stands_min_Z = 0
    i = 0
    for stand in stands[::2]:
        if stand.id in args.stand:
            print "----| %s" % str(stand)
            data[i,0] = stand.id
            data[i,1] = stand.x
            data[i,2] = stand.y
            data[i,3] = stand.z
            # Increase i if a used stand
            i += 1

        # Find y limits for plotting the wavefront vector
        if stand.y > wavefront_max_Y:
            wavefront_max_Y = stand.y
        elif stand.y < wavefront_min_Y:
            wavefront_min_Y = stand.y

    stands_max_X = max(data[:,1])
    stands_min_X = min(data[:,1])
    stands_max_Y = max(data[:,2])
    stands_min_Y = max(data[:,2])
    stands_max_Z = max(data[:,3])
    stands_min_Z = min(data[:,3])

    # Angle of arrival of wavefront (bearing from north, clockwise)
    tx=43.80805491642218
    tx_rad = math.radians(tx)
    if tx < 45 and tx > 0:
        wavefront_min_X = wavefront_min_Y*math.tan(tx_rad)
        wavefront_max_X = wavefront_max_Y*math.tan(tx_rad)
    elif tx > 45 and tx < 90:
        wavefront_min_Y = wavefront_min_X/math.tan(tx_rad)
        wavefront_max_Y = wavefront_max_X/math.tan(tx_rad)
    else:
        print "I CAN'T DO THIS THE WAVEFRONT ANGLE IS FUNKY"
    wavefront_start = (wavefront_min_X, wavefront_min_Y, 0)
    wavefront_end = (wavefront_max_X, wavefront_max_Y, 0)

    # Compute perpendicular distance to the waveform vector
    for i in range(len(data)):
        (data[i,4], (data[i,5], data[i,6], junk)) = lat.pnt2line((data[i,1],data[i,2], 0), wavefront_start, wavefront_end)
        i += 1



    # Color-code the stands by their elevation
    color = data[:,3]
    
    # Plot the stands as colored circles
    fig = plt.figure(figsize=(8,8))
    
    ax1 = plt.axes([0.30, 0.30, 0.60, 0.60])
    ax2 = plt.axes([0.30, 0.05, 0.60, 0.15])
    ax3 = plt.axes([0.05, 0.30, 0.15, 0.60])
    ax4 = plt.axes([0.05, 0.05, 0.15, 0.15])
    c = ax1.scatter(data[:,1], data[:,2], c=color, s=40.0, alpha=0.50)	
    ax1.set_xlabel('$\Delta$X [E-W; m]')
    ax1.set_ylabel('$\Delta$Y [N-S; m]')
    ax1.set_title('%s Site:  %.3f$^\circ$N, %.3f$^\circ$W' % (station.name, station.lat*180.0/np.pi, -station.long*180.0/np.pi))
    
    # Plot line perpendicular to wavefront
    ax1.plot([wavefront_min_X,wavefront_max_X], [wavefront_min_Y, wavefront_max_Y])
    
    ax2.scatter(data[:,1], data[:,3], c=color, s=40.0)
    ax2.xaxis.set_major_formatter( NullFormatter() )
    ax2.set_ylabel('$\Delta$Z [m]')
    ax3.scatter(data[:,3], data[:,2], c=color, s=40.0)
    ax3.yaxis.set_major_formatter( NullFormatter() )
    ax3.set_xlabel('$\Delta$Z [m]')
    
    # Mark and anotate all points
    for i in xrange(toMark.size):
        ax1.plot(data[i,1], data[i,2], marker='x', linestyle=' ', color='black')
        ax2.plot(data[i,1], data[i,3], marker='x', linestyle=' ', color='black')
        ax3.plot(data[i,3], data[i,2], marker='x', linestyle=' ', color='black')
        ax1.annotate('%i' % (data[i,0]), xy=(data[i,1:3]), xytext=(data[i,1]+2, data[i,2]-2*math.tan(tx_rad)))
        ax2.annotate('%0.2f' % (data[i,3]), xy=(data[i,1], data[i,3]), xytext=(data[i,1]+3, data[i,3]-0.2))
                
    # Add an elevation colorbar to the left-hand side of the figure
    cb = plt.colorbar(c, cax=ax4, orientation='vertical', ticks=[])#data[:,3])
    cb.set_ticks( [] )

    #Compute and plot distances between points along wavefront vector
    y_sorted_data = data[np.argsort(data[:,2])]
    print "--| Distance along wavefront vector from:"
    for i in range(len(y_sorted_data)-1):
        ref = -1
        ref_pt = (y_sorted_data[ref,1], y_sorted_data[ref,2], 0)
        ref_pt_projected_on_wavefront = (y_sorted_data[ref,5], y_sorted_data[ref,6], 0)
        sec = -2
        sec_pt = (y_sorted_data[sec,1], y_sorted_data[sec,2], 0)
        sec_pt_projected_on_wavefront = (y_sorted_data[sec,5], y_sorted_data[sec,6], 0)

        if i == 0: # first time only
            ax1.plot([ref_pt_projected_on_wavefront[0], ref_pt[0]],[ref_pt_projected_on_wavefront[1], ref_pt[1]], color='gray', linestyle='--')
        ax1.plot([sec_pt_projected_on_wavefront[0], sec_pt[0]],[sec_pt_projected_on_wavefront[1], sec_pt[1]], color='gray', linestyle='--')

        dist = lat.distance(ref_pt_projected_on_wavefront, sec_pt_projected_on_wavefront)
        mid_pt = lat.mid(ref_pt_projected_on_wavefront, sec_pt_projected_on_wavefront)
        
        ax1.annotate('%0.2f m' % dist, xy=mid_pt[:2], xytext=(mid_pt[0]+3*math.cos(tx_rad), mid_pt[1]-3*math.sin(tx_rad)))
        ax1.annotate("",
            xy=(ref_pt_projected_on_wavefront[:2]), xycoords='data',
            xytext=(sec_pt_projected_on_wavefront[:2]), textcoords='data',
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        print "----| %s to %s = %s" % (y_sorted_data[ref,0], y_sorted_data[sec, 0], dist)

        y_sorted_data = np.delete(y_sorted_data, ref, 0)
        i += 1

    
    # Set the axis limits
    ax1.set_xlim([-60, 60])
    ax1.set_ylim([-60, 60])
    ax2.set_xlim( ax1.get_xlim() )
    ax3.set_ylim( ax1.get_ylim() )
    
    # Show n' save
    if args.suppress is not None:
        plt.show()
    if args.output is not None:
        fig.savefig(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='plot the x, y, and z locations of stands at an LWA station', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('stand', type=int, nargs='*', 
                        help='stand number to mark')
    parser.add_argument('-s', '--lwasv', action='store_true', 
                        help='use LWA-SV instead of LWA1')
    parser.add_argument('-m', '--metadata', type=str, 
                        help='name of the SSMIF or metadata tarball file to use for mappings')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='run %(prog)s in verbose mode')
    parser.add_argument('-o', '--output', type=str, 
                        help='filename to save the plot to')
    parser.add_argument('-p', '--suppress', action='store_true',
                        help='suppress live plot')
    args = parser.parse_args()
    main(args)
    
