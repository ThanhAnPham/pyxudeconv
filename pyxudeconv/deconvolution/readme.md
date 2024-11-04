# Deconvolution with Pyxu

The main script `main_deconvolution.py` can be called as a command-line with arguments or via a bash file (see `main_example.sh` or `main_calibration.sh`).

Two arguments are important to run on your own data
- `datapath`: Path to the data to deconvolve
- `psfpath`: Path to the point-spread function

Currently supported file formats
- `.czi`: Carl Zeiss files
- `.tif`: Expected order of the dimension (Time, Views, Channels, Z, Y, X). Note that the file is first fully loaded, then the region of interest is kept for further processing. One drawback is that the RAM memory usage may be temporarily large.


An example of calling the script with a command-line

`python main_deconvolution.py --fres '../res/donuts' --gpu 0 --datapath '../data/real_donut/data.tif' --psfpath '../data/real_donut/psf.tif' --saveIter 10 10 10 10 10 --nviews 1 --methods 'RL' 'GARL' --Nepoch 50 --bufferwidth 20 10 10   --pxsz 79.4 79.4 1000 --bg 0 --psf_sz -1 -1 128 128 --roi 0 0 150 150 --config_GARL 'widefield_params'`

## Installation

If Goujon accelerated Richardon-Lucy (GARL) and/or GPU will be used, please install `torch`[^1] according to your case. For instance, If the GPU CUDA version is 12.1, the conda environment can be created in a terminal with the commands

- `conda create -n pyxudeconv python=3.11 pytorch=2.4.1 pytorch-cuda=12.1 tifffile numpy scipy matplotlib cupy -c pytorch -c nvidia -c conda-forge` 
- `conda activate pyxudeconv`
- `pip install pyxu`
<!--- `pip install git+https://github.com/pyxu-org/pyxu.git@feature/fast_fftconvolve pylibCZIrw`) --->

[^1]:21/10/2024, there might be an incompatiblity with the `sympy(==1.13.1)` package version required by `pytorch >= 2.5.0`. Either downgrade `sympy` to `1.13.1` (but may create incompatibilities with `pyxu`) or install `pytorch=2.4.1`.

## Goujon Accelerated Richardson-Lucy (GARL)

To use GARL, call `python main_deconvolution.py` with the argument `--methods 'GARL'`.
To run over different hyperparameters, you can create your own configuration file `your_config_file.py` in the folder `deconvolution/methods/configs/GARL/` and add the argument `--config_GARL 'your_config_file'`

## Simulation

The file `simulate.py`can simulate measurements obtained from a phantom defined by `--phantom your_phantom_file` convolved with a PSF defined by `--psfpath your_psf_file`. Future releases may change the organisation of the simulation part.