#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:06:20 2024

@author: jguerraa
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polBpy",
    version="0.1.1",
    author="Jordan Guerra",
    author_email="jordan.guerra@gmail.com",
    description="Package for performing angular dispersion analysis of polarimetric data\
        and DCF calculations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jorgueagui/polBpy",
    packages=setuptools.find_packages(include=['polBpy','polBpy.*']),
    install_requires=[
        'numpy',
        'matplotlib',
        'emcee',
        'george',
        'scipy',
        'joblib',
        'astropy'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)