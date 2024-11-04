import os, importlib
import numpy as np


try:
    import cupy as cp
except ImportError as e:
    import numpy as cp


def get_reader(fpath):
    r'''
    Load the correct file reader based on file extension
    '''
    if isinstance(fpath, str):
        f_extension = fpath[fpath.rfind('.') + 1:]
        if os.path.exists(f'./reader/read_{f_extension}.py'):
            module_read = importlib.import_module(f'reader.read_{f_extension}')
            read_file = getattr(module_read, f'read_{f_extension}')
            return read_file
        else:
            print(
                f'Unknown file format for file ({f_extension}).',
                'Please add a reader in reader folder or convert the file',
            )
            return 0
    elif isinstance(fpath, (np.ndarray, cp.ndarray)):

        def read_file(fpath, *args, **kwargs):
            return fpath

        return read_file
    else:
        print('Unkown format for psf path and data path or not ndarray')
        return 0
