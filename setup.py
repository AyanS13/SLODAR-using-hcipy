#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup, Extension
import numpy

slodar_module = Extension('slodar', sources = ['slodar.c'], include_dirs = [numpy.get_include()], libraries = ['gsl','gslcblas'],)

setup(name='slodar',
      version='0.1.0',
      description='C SLODAR module',
      ext_modules=[slodar_module])




