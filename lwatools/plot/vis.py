#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from lwatools.vis_modeling.visibility_models import point_source_visibility_model_uv

def get_vis_indices(id_pairs):
    indices = []
    for p in id_pairs:
        i = (k for k,b in enumerate(baseline_pairs) if b[0].id == p[0] and b[1].id == p[1])
        indices.append(next(i))

    return indices

def vis_phase_scatter_3d(u, v, vis, show=True, html_savename=None, title="", l=None, m=None, model=point_source_visibility_model_uv, cplx_map=np.angle):
    '''
    Saves a 3D scatter plot of visibilites. Pass l,m to add a model to the plot as well.
    '''
    u = u.flatten()
    v = v.flatten()
    vis = vis.flatten()

    data = []
    data.append(go.Scatter3d(x=u, y=v, z=cplx_map(vis), mode='markers', marker=dict(size=1, color='red')))

    if l and m:
        data.append(go.Scatter3d(x=u, y=v, z=cplx_map(point_source_visibility_model_uv(u, v, l, m)), mode='markers', marker=dict(size=1, color='black')))

    fig = go.Figure(data=data)

    fig.update_layout(scene=dict(
        xaxis_title='u',
        yaxis_title='v',
        zaxis_title='phase'),
        title=title)

    if html_savename:
        print(f"saving visibility phase scatter to {html_savename}")
        fig.write_html(html_savename)

    if show:
        print("showing visibility scatter")
        fig.show()
    

def plot_vis_2d(bl_array, visibilities, output_dir='.'):
    plt.close('all')
    for k, vis in enumerate(visibilities):
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].scatter(bl_array, abs(vis), color = 'black', marker='.', s=0.1)
        ax[0].set_ylabel("Visibility Magnitude")
        ax[0].set_ylim(0, 60000)
        ax[1].scatter(bl_array, np.angle(vis), color='black', marker='.', s=0.1)
        ax[1].set_ylabel("Visibility Phase")
        ax[1].set_xlabel("Projected Baseline")
        plt.savefig("{}/{}.png".format(output_dir, k))


def plot_unprojected(baseline_pairs, visibilities):
    # this plots magnitude and phase of visibility as a function of radial baseline from the center of the array
    bl = [np.sqrt((b[0].stand.x - b[1].stand.x)**2 + (b[0].stand.y - b[1].stand.y)**2) for b in baseline_pairs]

    plot_vis_2d(bl, visibilities)


def project_baselines(baseline_pairs, azimuth):
    # project all of the visibility measurements from the 2D baseline space on to a specific direction
    # imagine looking at the side of the array perpendicular to the direction of incidence
    unit = np.array([-np.cos(np.pi/2 - azimuth), -np.sin(np.pi/2 - azimuth)])

    bl = np.array([np.array([b[0].stand.x - b[1].stand.x, b[0].stand.y - b[1].stand.y]) for b in baseline_pairs])

    return np.dot(bl, unit)

def plot_projected(baseline_pairs, visibilities, azimuth, output_dir='.'):
    bl = project_baselines(baseline_pairs, azimuth)

    plot_vis_2d(bl, visibilities, output_dir)
