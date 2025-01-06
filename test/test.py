import pyxudeconv as pd
import unittest
import importlib
import os

#ToDo: check loading of methods. They require proper solver initialization and hyper-parameters (list/tuple, not scalar)

class TestMethod(unittest.TestCase):
    '''
    def test_methods(self):
        meths = os.path('pyxudeconv.deconvolution.methods')
        for cm in meths:
            mod = importlib.import_module(
                f'pyxudeconv.deconvolution.methods.{cm}')
    '''