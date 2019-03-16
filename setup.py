import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

ext_modules = [
    Extension(
        name='mtcnn.utils.cython_bbox',
        sources=['mtcnn/utils/cython_bbox.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name='mtcnn.utils.cython_nms',
        sources=['mtcnn/utils/cython_nms.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name='mtcnn',
    version='0.1',
    author='tkhe',
    packages=['mtcnn'],
    ext_modules=cythonize(ext_modules)
)
