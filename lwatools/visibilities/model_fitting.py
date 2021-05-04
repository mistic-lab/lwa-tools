from scipy.optimize import least_squares

from lwatools.file_tools.outputs import build_output_file
from lwatools.visibilities.models import point_residual_abs

def ls_cost(params, u, v, vis, resid=point_residual_abs):
    '''
    Computes the least-squares cost function at the given parameter values.
    least_squares actually takes care of this step, but this is here for
    visualization and debugging purposes.
    '''
    r = resid(params, u, v, vis)
    return np.dot(r,r)

def fit_model_to_vis(uvw, vis, residual_function, l_init, m_init,
        opt_method='lm', export_npy=False, param_guess_av_length=10, verbose=True):
    '''
    Fits a point source (or equivalently a gaussian) model to the visibilities in vis.

    uvw should be a (len(vis), 3) array of baseline vectors corresponding to
    the visibiltiy samples.

    It's monochromatic (single-frequency) for now.

    l_est, m_est should be arrays of previous l,m values, the mean of which is
    used as an optimization starting point.

    returns l, m, opt_result
    
    The optimized l,m parameter values and full optimization result dictionary.
    '''

    # monochromatic for now
    # TODO: make it not monochromatic

    # we're only fitting the phase, so normalize the visibilities
    #vis = vis/np.abs(vis)

    u = uvw[:, 0]
    v = uvw[:, 1]
    w = uvw[:, 2]

    if export_npy:
        print("Exporting u, v, w, and visibility")
        np.save('u{}.npy'.format(k), u)
        np.save('v{}.npy'.format(k), v)
        np.save('w{}.npy'.format(k), w)
        np.save('vis{}.npy'.format(k), vis)


    if verbose:
        print("Optimizing")

    opt_result = least_squares(
            residual_function,
            [l_init, m_init], 
            args=(u, v, w, vis),
            method=opt_method
            )

    if verbose:
        print("Optimization result: {}".format(opt_result))
        print("Start point: {}".format((l_init, m_init)))
    l_out, m_out = opt_result['x']
    
    return l_out, m_out, opt_result

