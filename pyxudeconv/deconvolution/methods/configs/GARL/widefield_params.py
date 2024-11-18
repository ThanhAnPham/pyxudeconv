import os
import inspect
import pyxudeconv

package_dir = os.path.dirname(inspect.getfile(pyxudeconv))

def widefield_params():
    params = dict()
    params['model'] = [
        os.path.join(package_dir, 'trained_models','3Dtubes')
    ]
    params['epochoi'] = [40180]
    params['lmbd'] = [0.1,0.5]  #np.arange(0.6, 2.1, 0.2)
    params['sigma'] = [0.1,0.5]  #np.arange(0.1, 2.1, 0.2)
    return params