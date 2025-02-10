__version__ = "0.0.2.8.1"

from pyxudeconv.deconvolution.deconvolve import deconvolve
from pyxudeconv.deconvolution.params import get_param
from pyxudeconv.deconvolution.simulate import simulate

__all__ = ['deconvolve','get_param','simulate']