#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:18:53 2018

@author: Sven serneels, Ponalytics
"""

from setuptools import setup, find_packages
import re
import os


def get_version():
    """Read version from __init__.py without importing the module."""
    init_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "direpack", "__init__.py"
    )
    with open(init_file, "r") as f:
        content = f.read()
    version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_metadata():
    """Read author and license from __init__.py without importing."""
    init_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "direpack", "__init__.py"
    )
    with open(init_file, "r") as f:
        content = f.read()
    author_match = re.search(r'^__author__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    license_match = re.search(r'^__license__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return (
        author_match.group(1) if author_match else "Unknown",
        license_match.group(1) if license_match else "MIT"
    )


__version__ = get_version()
__author__, __license__ = get_metadata()

readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed in user environment
    with open(readme_file) as f:
        readme = f.read()

setup(
    name="direpack",
    version=__version__,
    author=__author__,
    author_email="svenserneels@gmail.com",
    description="A Python 3 Library for State-of-the-Art Statistical Dimension Reduction Methods",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/SvenSerneels/direpack",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages('src'),  # include all packages under src
    package_dir={'':'src'},   # tell distutils packages are under src
    include_package_data = True,
    install_requires=[
        'numpy>=1.5.0',
        'scipy>=0.8.0',
        'matplotlib>=2.2.0',
        'scikit-learn>=0.18.0',
        'pandas>=0.19.0',
        'statsmodels>=0.8.0',
        # 'ipopt>=0.1.5',
        'dcor>=0.3'
    ]
)

