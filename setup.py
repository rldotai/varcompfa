#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# To use a consistent encoding
import codecs
from setuptools import setup, find_packages
import sys, os.path


def parse_reqs(req_path='./requirements.txt'):
    """Recursively parse requirements from nested pip files."""
    install_requires = []
    with codecs.open(req_path, 'r') as handle:
        # remove comments and empty lines
        lines = (line.strip() for line in handle
                 if line.strip() and not line.startswith('#'))
        for line in lines:
            # check for nested requirements files
            if line.startswith('-r'):
                # recursively call this function
                install_requires += parse_reqs(req_path=line[3:])
            else:
                # add the line as a new requirement
                install_requires.append(line)
    return install_requires

setup(
    name='varcompfa',
    version='0.7.1',
    description='reinforcement learning variance algorithm comparisons',
    url='https://bitbucket.org/adaptiveprostheticsgroup/varcompfa',
    author='adaptiveprostheticsgroup',
    author_email='',
    license='',
    packages=[package for package in find_packages(exclude=('tests*', 'docs*'))
        if package.startswith('varcompfa')],
    zip_safe=False,
    # Install requirements loaded from ``requirements.txt``
    install_requires=parse_reqs()
)

