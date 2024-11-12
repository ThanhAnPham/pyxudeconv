"""
This file allows a user to call deconvolve from terminal as ```pyxudeconv.deconvolve --[parameters]``` without going too deep in the file hierarchy.
"""

from .deconvolution.deconvolve import deconvolve
deconvolve()