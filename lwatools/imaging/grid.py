import numpy as np

def gauss_kernel_1d(offset, extent, sigma=1):
    '''
    computes a sampled gaussian kernel with an offset

    offset should be between -1 and 1 and indicates how far the peak of the
    kernel should be offset from the middle of the array

    returns a numpy array with the kernel samples. the middle of the kernel is at 
    index extent//2
    '''
    x = np.arange(extent) - extent//2
    g = 1/(sigma * np.sqrt(np.pi)) * np.exp(-0.5 * (x - offset)**2 / sigma**2)
    return g

def grid_point_with_offset(u, v, w, vis, kernel_1d, kernel_extent, res):
    '''
    this one isn't tested!
    '''

    # Find the grid point closest to the sample we're gridding.
    # This is where the kernel will be centered.
    # The offset shifts the kernel to correct for the fact that our point
    # probably isn't exactly on the grid.
    center_u = np.round(u/res)
    offset_u = u - center_u * res
    center_v = np.round(v/res)
    offset_v = v - center_v * res

    # compute the two parts of the kernel
    ku = kernel_1d(offset_u, kernel_extent)
    kv = kernel_1d(offset_v, kernel_extent)

    # the 2D kernel at each point is the product of the two 1d kernels
    # we can compute this with the outer product B)
    out = np.matmul(kv, ku.T)

    # scale by the visibility
    out *= vis 

    return out, center_u, center_v
    
def grid_vis(uvw, vis, size=80, res=0.5, extent=7, sigma=0.5):
    '''
    note: neglects W coordinate completely - shouldn't be a problem, right?
    '''

    # create the grid and an array for the gridded visibilities
    gru, grv = np.mgrid[-size/2:size/2:res, -size/2:size/2:res]
    grvis = np.zeros_like(gru, dtype=np.complex)

    # separate out the coordinates of the baseline-sampled points
    blu = uvw[:, 0]
    blv = uvw[:, 1]
    blw = uvw[:, 2]

    # generate the 1d gaussian kernel which will be used to find the 2d kernel points
    gk1d = gauss_kernel_1d(0, extent, sigma=sigma)
    
    # the 2d gaussian kernel at (i,j) is gk1d(i) * gk2d(j), we can use the outer product
    gk1d.shape = (gk1d.shape[0], 1)
    gk2d = np.matmul(gk1d, gk1d.T)

    for k in range(len(vis)):
        uk = uvw[k, 0]
        vk = uvw[k, 1]
        visk = vis[k]
        center_u = int(np.round(uk/res) + size/(2*res) )
        center_v = int(np.round(vk/res) + size/(2*res) )

        grvis[center_u - extent//2 : center_u + extent//2 + 1, center_v - extent//2 : center_v + extent//2 + 1] += gk2d * visk

    return gru, grv, grvis
