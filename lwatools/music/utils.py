'''
Utilities for the spatial MUSIC algorithm. This was implemented primarily as a
comparison for the visibility-modelling DoA estimation methods. It's quite slow.
'''
import numpy as np
import aipy

def compute_acm(data, freq, valid_ants, verbose=False):
    idx = [ant.digitizer - 1 for ant in valid_ants]
    xyz = np.array([[ant.stand.x, ant.stand.y, ant.stand.z] for ant in valid_ants])

    delays = np.array([ant.cable.delay(freq) for ant in valid_ants])
    delays -= delays.min()

    data = data[idx, :]

    # apply the phase rotation to deal with the cable delays
    if verbose:
        print("Correcting for cable delays")
    for i in range(data.shape[0]):
        data[i, :] *= np.exp(2j*np.pi*freq*delays[i])
    data /= (np.abs(data)).max()

    if verbose:
        print("Computing ACM")
    nSamp = data.shape[1]
    for i in range(nSamp):
        x = np.matrix(data[:,i]).T
        try:
            acm += x * x.H
        except:
            acm = x * x.H

    acm /= nSamp

    return acm, xyz

def generate_acm_from_tbn(tbnf, valid_ants, integration_length):
    freq = tbnf.get_info('freq1')

    n_samples = tbnf.get_info('nframe') / tbnf.get_info('nantenna')
    samples_per_integration = int(integration_length * tbnf.get_info('sample_rate')/512)
    n_integrations = int(np.floor(n_samples / samples_per_integration))

    for int_num in range(n_integrations):
        tint, t0, data = tbnf.read(integration_length)
        
        acm, xyz = compute_acm(data, freq, valid_ants)        

        yield acm, xyz, freq
        

def compute_music_spectrum(acm, xyz, freq, n_points=100, verbose=False):
    '''
    Evaluates the MUSIC pseudospectrum at equally-spaced points 
    '''

    img = aipy.img.ImgW(size=n_points//2, res=0.5)
    top = img.get_top(center=(n_points//2, n_points//2))
    saz, sel = aipy.coord.top2azalt(top)

    if verbose:
        print("Computing eigenvectors/values of the ACM")

    evals, evecs = np.linalg.eig(acm)
    order = np.argsort(np.abs(evals))[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # only looking for a single source
    Usig_idx = range(1)
    Unoise_idx = range(1, evals.size)

    if verbose:
        print("Evaluating the MUSIC pseudospectrum")

    spectrum = np.zeros_like(saz)

    for i in range(saz.shape[0]):
        if verbose:
            print(f"Computing grid row {i+1} / {saz.shape[0]}")
        for j in range(saz.shape[1]):
            az = saz[i,j]
            el = sel[i,j]
            if not np.isfinite(az) or not np.isfinite(el):
                continue

            uvec = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
            steering_vector = np.zeros((len(xyz), 1), dtype=np.complex128)
            for k in range(len(xyz)):
                steering_vector[k, 0] = np.exp(2j * np.pi * freq * np.dot(xyz[k,:] - xyz[0,:], uvec)/3e8)

            steering_vector = np.matrix(steering_vector)

            noise_evecs = np.matrix(evecs[:, Unoise_idx])

            denom = steering_vector.H * noise_evecs * noise_evecs.H * steering_vector

            spectrum[i, j] = 1.0/max([1e-9, denom[0,0].real])

    return saz, sel, spectrum
