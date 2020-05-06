#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def requirements():
    with open("requirements.txt", "r+") as f:
        return f.read()


def long_description() -> str:
    # Read the contents of README file
    this_directory = os.path.dirname(__file__)
    with open(os.path.join(this_directory, "README.rst"), encoding="utf-8") as f:
        readme = f.read()
    return readme


setup(
    name='napari-mat-images',
    version='0.1.1',
    author='Hector Munoz',
    author_email='hectormz.git@gmail.com',
    maintainer='Hector Munoz',
    maintainer_email='hectormz.git@gmail.com',
    license='BSD-3',
    url='https://github.com/hectormz/napari-mat-images',
    description='A plugin to load images stored in .mat files with napari',
    long_description=long_description(),
    long_description_content_type='text/x-rst',
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
    entry_points={'napari.plugin': ['mat-images = napari_mat_images']},
)
