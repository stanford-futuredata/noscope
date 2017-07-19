#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from os.path import join as pjoin
import os
import glob
import subprocess


setup(name='noscope',
      version='0.0.1',
      description='',
      author='Daniel Kang',
      author_email='ddkang@stanford.edu',
      packages=['noscope'],
      install_requires=open('requirements.txt').read().split('\n'))
