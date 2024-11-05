import numpy as np
from tifffile import TiffFile


def read_tif(
        filepath,
        Hoi=None,
        Zoi=None,
        Coi=None,
        Toi=None,
        roi=(None, None, None, None),
        zoom=1,
):
    r'''Read (ome.)tif files. Expects a 3D stack (volume) at least.
    Can only be up to 5D (TCZYX) where T may encode several views (e.g., like in Airyscan).
    Note that tifffile package loads the hyperstack as TCZYX (if the metadata is properly set).
    - The key 'slices' in imagej_metadata dictionary relates to T (Hoi)
    - The key 'channels' in imagej_metadata dictionary relates to C (Coi)
    - The key 'frame' in imagej_metadata dictionary relates to Z (Zoi)
    
    Note 2: if T and C are both singleton dimensions, the metadata key 'slices' may refers to 'Z'.
    '''
    with TiffFile(filepath) as tif:
        data = tif.asarray()
        data = np.expand_dims(data, axis=tuple(np.arange(5 - data.ndim)))
        imagej_metadata = tif.imagej_metadata
        if Hoi is None:
            if data.squeeze().ndim == 3:
                Hoi = 0
            else:
                if 'slices' in imagej_metadata.keys():
                    Hoi = np.arange(0, imagej_metadata['slices'])
                else:
                    Hoi = 0
            #Hoi = np.arange(0, data.shape[-5])
        Hoi = np.array(Hoi)
        if Coi is None:
            if 'channels' in imagej_metadata.keys():
                Coi = np.arange(0, imagej_metadata['channels'])
            else:
                Coi = 0
        Coi = np.array(Coi)
        if Zoi is None:
            Zoi = np.arange(0, data.shape[-3])
        Zoi = np.array(Zoi)

        roi = np.array(roi)

        if np.any(roi[2:]==None) or np.any(roi[2:] <= 0):
            roi = np.array((0, 0, *data.shape[-2:]))
        elif np.any(roi[:2]==None) or np.any(roi[:2] < 0):
            # No negative indices
            roi[0] = np.maximum(data.shape[-2] // 2 - roi[2] // 2, 0)
            roi[1] = np.maximum(data.shape[-1] // 2 - roi[3] // 2, 0)
        #make sure that roi doesn't go out of bounds
        roi[-2:] = np.minimum(roi[:2] + roi[-2:] - 1,
                              np.array(data.shape[-2:]) - 1) - roi[:2] + 1
        roi = tuple(map(int, roi))
        out = data[
            Coi.reshape((-1, 1, 1, 1, 1)),
            Hoi.reshape((1, -1, 1, 1, 1)),
            Zoi.reshape((1, 1, -1, 1, 1)),
            roi[0]:roi[0] + roi[2],
            roi[1]:roi[1] + roi[3],
        ].astype('float32').squeeze()
        return out
