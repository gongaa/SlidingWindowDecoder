from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

VERSION="0.0.0"
f = open("src/VERSION","w+")
f.write(VERSION)
f.close()

extension1 = Extension(
    name="src.mod2sparse",
    sources=["src/include/mod2sparse.c", "src/mod2sparse.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), 'src/include'],
    extra_compile_args=['-std=c11']
    )

extension2 = Extension(
    name="src.c_util",
    sources=["src/c_util.pyx", "src/include/mod2sparse.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), 'src/include'],
    extra_compile_args=['-std=c11']
    )

extension3 = Extension(
    name="src.bp_guessing_decoder",
    sources=["src/bp_guessing_decoder.pyx", "src/include/mod2sparse.c", "src/include/bpgd.cpp"],
    language='c++',
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), 'src/include'],
    extra_compile_args=['-std=c11']
    )

extension4 = Extension(
    name="src.osd_window",
    sources=["src/osd_window.pyx", "src/include/mod2sparse.c", "src/include/mod2sparse_extra.cpp", "src/include/bpgd.cpp"],
    language='c++',
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), 'src/include'],
    extra_compile_args=['-std=c11']
)

extension5 = Extension(
    name="src.bp4_osd",
    sources=["src/bp4_osd.pyx", "src/include/mod2sparse.c", "src/include/mod2sparse_extra.cpp", "src/include/bpgd.cpp"],
    language='c++',
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), 'src/include'],
    extra_compile_args=['-std=c11']
)

setup(
    version=VERSION,
    ext_modules=cythonize([extension1, extension2, extension3, extension4, extension5]),
)
