from distutils.core import setup
from Cython.Build import cythonize

setup(name='pretrain_model', ext_modules=cythonize('pretrain.pyx'))