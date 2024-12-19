import numpy as np
from pylibCZIrw import czi


def read_czi(
        filepath,
        Hoi=None,
        Zoi=None,
        Coi=None,
        Toi=None,
        roi=(None, None, None, None),
        zoom=1,
):
    '''Read CZI files'''

    with czi.open_czi(filepath) as czidoc:
        # get the image dimensions as a dictionary, where the key identifies the dimension
        total_bounding_box = czidoc.total_bounding_box

        # get the total bounding box for all scenes
        total_bounding_rectangle = np.array(czidoc.total_bounding_rectangle)

        # get the bounding boxes for each individual scene
        scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle
        if Hoi is None:
            Hoi = np.array(total_bounding_box['H'])
            Hoi = np.arange(Hoi[0], Hoi[-1])
        Hoi = np.array(Hoi)

        Hoi = Hoi[Hoi < total_bounding_box['H']
                  [-1]]  #assumes views always start at 0
        if Zoi is None:
            Zoi = np.array(total_bounding_box['Z'])
            Zoi = np.arange(Zoi[0], Zoi[-1])
        Zoi = np.array(Zoi)
        if Coi is None:
            Coi = np.array(total_bounding_box['C'])
            Coi = np.arange(Coi[0], Coi[-1])
        Coi = np.array(Coi)
        if Toi is None:
            Toi = np.array(total_bounding_box['T'])
            Toi = np.arange(Toi[0], Toi[-1])

        roi = np.array(roi)
        if np.any(roi[2:]==None) or np.any([cr <= 0 for cr in roi[2:]]):
            roi = total_bounding_rectangle
        elif np.any(roi[:2]==None) or np.any([cr < 0 for cr in roi[:2]]):
            roi[0] = np.maximum(
                (total_bounding_rectangle[2] - total_bounding_rectangle[0]) //
                2 - roi[2] // 2, np.zeros(1))
            roi[1] = np.maximum(
                (total_bounding_rectangle[3] - total_bounding_rectangle[1]) //
                2 - roi[3] // 2, np.zeros(1))
        #makes sure that roi doesn't go out of bounds

        roi[-2:] = np.minimum(roi[:2] + roi[-2:] - 1,
                              total_bounding_rectangle[-2:] - 1) - roi[:2] + 1

        roi = tuple(map(int, roi))
        out = np.zeros((np.size(Toi), np.size(Hoi), np.size(Coi), np.size(Zoi),
                        roi[-2], roi[-1]))
        for ziter in np.arange(np.size(Zoi)):
            for citer in np.arange(np.size(Coi)):
                for hiter in np.arange(np.size(Hoi)):
                    for titer in np.arange(np.size(Toi)):
                        hc = Hoi[hiter] if np.ndim(Hoi) > 0 else Hoi
                        cc = Coi[citer] if np.ndim(Coi) > 0 else Coi
                        zc = Zoi[ziter] if np.ndim(Zoi) > 0 else Zoi
                        tc = Toi[titer] if np.ndim(Toi) > 0 else Toi
                        out[titer, hiter, citer, ziter] = np.squeeze(
                            czidoc.read(plane={
                                "H": hc,
                                "Z": zc,
                                "C": cc,
                                "T": tc
                            },
                                        zoom=zoom,
                                        roi=roi))
        return np.squeeze(out)
