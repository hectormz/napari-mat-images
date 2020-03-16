#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def requirements():
    with open("requirements.txt", "r+") as f:
        return f.read()


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


setup(
    name='napari-mat-images',
    version='0.1.0',
    author='Hector Munoz',
    author_email='hectormz.git@gmail.com',
    maintainer='Hector Munoz',
    maintainer_email='hectormz.git@gmail.com',
    license='BSD-3',
    url='https://github.com/hectormz/napari-mat-images',
    description='A plugin to load images stored in .mat files with napari',
    long_description=read('README.rst'),
    py_modules=['napari_mat_images'],
    python_requires='>=3.6',
    install_requires=requirements(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
    ],
    entry_points={'napari.plugin': ['mat-images = napari_mat_images',],},
)
