#!/usr/bin/python3

import numpy as np

def point_source_visibility_model_uvw(u, v, w, l, m):
    '''
    Computes visibility at u,v as if from a perfect point source located at (l,m)
    '''
    return np.exp(2j * np.pi * ( l*u + v*m + w*np.sqrt(1-l**2-m**2) ))

def point_source_visibility_model_uv(u, v, l, m):
    '''
    Computes visibility at u,v as if from a perfect point source located at (l,m)
    '''
    return np.exp(2j * np.pi * ( l*u + v*m))

def point_residual_cplx(params, u, v, vis):
    '''
    Calculates the residuals of the model fit with the differences of the real
    and imaginary parts each giving one residual.

    Parameters are l,m from point_source_visibility_model
    '''
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model_uv(u, v, w, l, m)

    return np.concatenate([mc.real, mc.imag]) - np.concatenate([vis.real, vis.imag])


def point_residual_abs(params, u, v, w, vis):
    '''
    Calculates the residual of the model fit as the magnitude of the difference
    between the model and the actual visibilities.

    Parameters are l,m from point_source_visibility_model
    '''
    l = params[0]
    m = params[1]
    # mc = point_source_visibility_model_uvw(u, v, w, l, m)
    mc = point_source_visibility_model_uv(u, v, l, m)

    return np.abs(mc - vis)

def gaussian_source_visibility_model(u, v, l, m, a):
    '''
    Computes the visibility at u,v as if from a gaussian source located at (l,m)

    a is a (scalar) width parameter. FWHM = np.sqrt(8 np.ln(2)) * a
    '''
    return np.exp(-2 * np.pi**2 * a**2 * (u**2 + v**2) + 2j * np.pi * (l*u + v*m))

def gaussian_residual_abs(params, u, v, w, vis, a=0.5):
    '''
    Calculates the residual of the model fit as the magintude of the difference
    between the model and the actual visibilities.

    Parameters are l,m from gaussian_source_visibility_model. The width parameter is fixed.
    '''
    l = params[0]
    m = params[1]
    mc = gaussian_source_visibility_model(u, v, l, m, a)

    return np.abs(mc - vis)


def bind_gaussian_residual(a):

    return lambda params, u, v, w, vis: gaussian_residual_abs(params, u, v, w, vis, a=a)
