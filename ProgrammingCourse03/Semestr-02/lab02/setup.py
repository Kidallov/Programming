'''
Файл для внедрения cython в python файл
'''

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("convolution_cython.pyx"),
    include_dirs=[np.get_include()]
)