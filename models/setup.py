from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='pretrain_model', ext_modules=cythonize('pretrain.pyx'), include_dirs=[np.get_include()])
