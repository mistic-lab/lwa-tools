import numpy as np

def point_source_visibility_model(u, v, l, m):
    '''
    Computes visibility at u,v as if from a perfect point source located at (l,m)
    '''
    return np.exp(2j * np.pi * (l*u + v*m))

def point_residual_cplx(params, u, v, vis):
    '''
    Calculates the residuals of the model fit with the differences of the real
    and imaginary parts each giving one residual.

    Parameters are l,m from point_source_visibility_model
    '''
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model(u, v, l, m)

    return np.concatenate([mc.real, mc.imag]) - np.concatenate([vis.real, vis.imag])


def point_residual_abs(params, u, v, vis):
    '''
    Calculates the residual of the model fit as the magnitude of the difference
    between the model and the actual visibilities.

    Parameters are l,m from point_source_visibility_model
    '''
    l = params[0]
    m = params[1]
    mc = point_source_visibility_model(u, v, l, m)

    return np.abs(mc - vis)
