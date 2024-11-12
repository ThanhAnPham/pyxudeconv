## Build operatorss for deconvolution

import pyxu.operator as pxo
from pyxu.operator import FFTConvolve
import numpy as np


def getModel(
    psfpath,
    datapath,
    psf_roi,
    nviews,
    coi_psf,
    roi,
    coi,
    bufferwidth,
    phantom,
    fid,
    xp,
    read_psf,
    read_data,
    normalize_meas=True,
):
    ''' Function that returns a phantom (if available), forward model, and the measurements
    For deconvolution, the PSF and data readers are expected to output a hyper-stack with dimensions order (Time x nviews x nchannels x Z x Y x X). If not relevant (e.g., Time), there is no need to create singleton dimension.
    '''
    hoi = tuple(range(nviews))
    psf = xp.array(
        read_psf(psfpath, Hoi=hoi, Coi=coi_psf, roi=psf_roi).astype(
            "float32"))  #Time, nviews, nchannels, z-axis, y-axis, x-axis #

    g = xp.array(
        read_data(datapath, Hoi=hoi, Coi=coi, roi=roi).astype(
            "float32"))  #(Time), nviews, nchannels, z-axis, y-axis, x-axis
    if psf.ndim > 3 + 1 * (isinstance(coi, tuple)):
        #Deconvolution problem has more than one view (Airyscan-like)
        if psf.shape[0] == 1:
            psf = psf.squeeze(0)
        nviews = psf.shape[
            0]  #data might have less views than the expected number set by parameters
        axoi = (0, -3, -2, -1)
    else:
        axoi = (-3, -2, -1)

    # Normalize

    psf /= psf.sum(
        axis=(-3, -2, -1),
        keepdims=True)  # Each view and each channel integrates to 1. (2,3,4)
    if normalize_meas:
        gnormalizer = g.max(axis=axoi, keepdims=True)  # (0,-3,-2,-1)
        g /= gnormalizer
        gnormalizer = gnormalizer.squeeze()
    else:
        gnormalizer = xp.ones(1)
    psf = psf.squeeze()
    g = np.maximum(g.squeeze(), 0)

    print('PSF shape', psf.shape, 'and min value', psf.min())
    # Define shape of the reconstructed image/volume
    padw = (*tuple([0] * (g.ndim - 3)), *bufferwidth)
    print('Measurements shape', g.shape, 'and min value', g.min())
    recon_shape = tuple(np.add(g.shape[-3:], np.array(padw[-3:]) * 2))
    pad_meas = pxo.Pad(g.shape, padw)

    #Physical model
    if nviews > 1:
        forw = pad_meas.T * pxo.stack([
            FFTConvolve(
                dim_shape=recon_shape,
                kernel=psf_view,
                center=tuple(np.array(psf_view.shape) // 2),
                mode='constant',
            ) for psf_view in psf
        ])
    else:
        forw = pad_meas.T * FFTConvolve(
            dim_shape=recon_shape,
            kernel=psf,
            center=tuple(np.array(psf.shape) // 2),
            mode='constant',
        )
    forw.lipschitz = xp.sum(
        xp.amax(xp.abs(xp.fft.fftn(psf, axes=(-3, -2, -1))),
                axis=(-3, -2, -1)))
    if phantom is not None:
        if None not in roi:
            # Crop phantom accordingly
            if phantom.ndim == 2:
                phantom = phantom[roi[1]:roi[1] + roi[3],
                                  roi[0]:roi[0] + roi[2]]
                phantom /= phantom.max()
            elif phantom.ndim == 3:
                phantom = phantom[:, roi[1]:roi[1] + roi[3],
                                  roi[0]:roi[0] + roi[2]]
                phantom /= phantom.max()
            if phantom.ndim == 4:
                phantom = phantom[:, coi, roi[1]:roi[1] + roi[3],
                                  roi[0]:roi[0] + roi[2]]
        #phantom /= phantom.max()
    trim_buffer = pxo.Trim(
        recon_shape, (*tuple([0] * (len(recon_shape) - 3)), *bufferwidth))
    if phantom is not None and phantom.ndim == 2:

        def op4metrics(x):
            return xp.max(
                pxo.Trim(recon_shape,
                         (*tuple([0] *
                                 (len(recon_shape) - 3)), *bufferwidth))(x),
                axis=-3,
            )
    else:
        op4metrics = trim_buffer

    fid = '{}_nv_{:d}_coi'.format(fid, nviews)
    if isinstance(coi, int):
        fid = '{}_{:d}'.format(fid, coi)
    else:
        for ccoi in coi:
            fid = '{}_{:d}'.format(fid, ccoi)

    return forw, g, trim_buffer, op4metrics, phantom, fid, psf, gnormalizer
