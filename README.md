# pyxudeconv

[![License MIT](https://img.shields.io/pypi/l/pyxudeconv.svg?color=green)](https://github.com/ThanhAnPham/pyxudeconv/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyxudeconv.svg?color=green)](https://pypi.org/project/pyxudeconv)
[![Python Version](https://img.shields.io/pypi/pyversions/pyxudeconv.svg?color=green)](https://python.org)
[![tests](https://github.com/ThanhAnPham/pyxudeconv/workflows/tests/badge.svg)](https://github.com/ThanhAnPham/pyxudeconv/actions)
[![codecov](https://codecov.io/gh/ThanhAnPham/pyxudeconv/branch/main/graph/badge.svg)](https://codecov.io/gh/ThanhAnPham/pyxudeconv)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/pyxudeconv)](https://napari-hub.org/plugins/pyxudeconv)

3D Deconvolution with Pyxu library

----------------------------------

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `pyxudeconv` via [pip]:

    pip install pyxudeconv

To install latest development version :

    pip install git+https://github.com/ThanhAnPham/pyxudeconv.git


## Deconvolution with Pyxu

### Deconvolving in a python code

After the package import, the deconvolution is performed by the function `deconvolve` which expects the parameters (namespace). To modify the parameters, there are two ways
  - Load the default parameters via `get_param` and modify each field of interest
    ````
    import pyxudeconv
    par = pyxudeconv.get_param()
    par.psfpath = '/home/tampham/3DWCR/data/simulated/psf_sample_calib_nv_32_coi_2.ome.tif'
    par.datapath = '/home/tampham/3DWCR/data/simulated/g_sample_calib_nv_32_coi_2.ome.tif'
    par.phantom = '/home/tampham/3DWCR/data/simulated/phantom_sample_calib_nv_32_coi_2.ome.tif'
    par.fres = '/home/tampham/yo'
    par.saveIter = [10]
    par.methods = ['RL','GARL','Tikhonov']
    imdeconv = pyxudeconv.deconvolve(par)
    ````
 - Change the json file and load it.
    ````
    import pyxudeconv
    par = pyxudeconv.get_param(param_file='./my_params.json')
    imdeconv = pyxudeconv.deconvolve(par)
    ````

Note that `par.psfpath`and `par.datapath` can be `numpy.ndarray` already loaded in the python code
````
par.psfpath = mypsf #numpy.ndarray
par.datapath = mydata #numpy.ndarray
````

### Deconvolving in a terminal
The main function `deconvolve` can be called as a command-line with arguments or via a bash file (see `main_example.sh` or `main_calibration.sh`) with the option -m.

Two arguments are important if applied on your own data
- `datapath`: Path to the data to deconvolve OR if ran through a python script it can be a ndarray itself
- `psfpath`: Path to the point-spread function OR if ran through a python script it can be a ndarray itself

Currently supported file formats
- `.czi`: Carl Zeiss files
- `.tif`: Expected order of the dimension (Time, Views, Channels, Z, Y, X). Note that the file is first fully loaded, then the region of interest is kept for further processing. One drawback is that the RAM memory usage may be temporarily large.

An example of calling the script with a command-line

```
python -m pyxudeconv.deconvolve --fres '../res/donuts' --gpu 0 --datapath '../data/real_donut/data.tif' --psfpath '../data/real_donut/psf.tif' --saveIter 10 10 10 10 10 --nviews 1 --methods 'RL' 'GARL' --Nepoch 50 --bufferwidth 20 10 10   --pxsz 79.4 79.4 1000 --bg 0 --psf_sz -1 -1 128 128 --roi 0 0 150 150 --config_GARL 'widefield_params'
```

## Note on dependencies

If Goujon accelerated Richardon-Lucy (GARL) and/or GPU will be used, please install `torch`[^1] according to your case. For instance, If the GPU CUDA version is 12.1, the conda environment can be created in a terminal with the commands

- `conda create -n pyxudeconv python=3.11 pytorch=2.4.1 pytorch-cuda=12.1 tifffile numpy scipy matplotlib -c pytorch -c nvidia -c conda-forge` 
- `conda activate pyxudeconv`
- `pip install pyxu[complete]`
<!--- `pip install git+https://github.com/pyxu-org/pyxu.git@feature/fast_fftconvolve pylibCZIrw`) --->

[^1]:21/10/2024, there might be an incompatiblity with the `sympy(==1.13.1)` package version required by `pytorch >= 2.5.0`. Either downgrade `sympy` to `1.13.1` (but may create incompatibilities with `pyxu`) or install `pytorch=2.4.1`.

## Goujon Accelerated Richardson-Lucy (GARL)

To use GARL, call `python -m pyxudeconv.deconvolve` with the argument `--methods 'GARL'`.
To run over different hyperparameters, you can add the argument `--config_GARL 'full_path/your_config_file.json'`.

Note: Each parameter must be a list of values, even if it is a single-valued list.
For instance, here is an example of a `.json` config file
````
{
    "WCRnet": ["pyxudeconv/trained_models/3Dtubes/"],
    "epochoi": [40180],
    "lmbd": [0.1, 0.5],
    "sigWC": [0.1, 0.5]
}
````

Alternatively, one can set a range of values for a parameter (e.g., `lmbd`) as follows
````
{
    "WCRnet": ["pyxudeconv/trained_models/3Dtubes/"],
    "epochoi": [40180],
    "lmbd_min": 0.1,
    "lmbd_max": 0.5,
    "lmbd_nsteps": 2,
    "sigWC": [0.1, 0.5]
}
````

## Simulation
#ToDo
The function `simulate`can simulate measurements obtained from a phantom defined by `--phantom your_phantom_file` convolved with a PSF defined by `--psfpath your_psf_file`. Future releases may change the organisation of the simulation part.


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"pyxudeconv" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/ThanhAnPham/pyxudeconv/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
