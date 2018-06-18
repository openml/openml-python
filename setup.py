# -*- coding: utf-8 -*-

import setuptools
import sys

with open("openml/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

dependency_links = []

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setuptools.setup(name="openml",
                 author="Matthias Feurer, Andreas Müller, Farzan Majdani, "
                        "Joaquin Vanschoren, Jan van Rijn and Pieter Gijsbers",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 maintainer="Matthias Feurer",
                 maintainer_email="feurerm@informatik.uni-freiburg.de",
                 description="Python API for OpenML",
                 license="BSD 3-clause",
                 url="http://openml.org/",
                 version=version,
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 install_requires=[
                     'mock',
                     'numpy>=1.6.2',
                     'scipy>=0.13.3',
                     'liac-arff>=2.2.1',
                     'xmltodict',
                     'nose',
                     'requests',
                     'scikit-learn>=0.18',
                     'nbformat',
                     'python-dateutil',
                     'oslo.concurrency',
                 ],
                 extras_require={
                     'test': [
                         'nbconvert',
                         'jupyter_client'
                     ]
                 },
                 test_suite="nose.collector",
                 classifiers=['Intended Audience :: Science/Research',
                              'Intended Audience :: Developers',
                              'License :: OSI Approved :: BSD License',
                              'Programming Language :: Python',
                              'Topic :: Software Development',
                              'Topic :: Scientific/Engineering',
                              'Operating System :: POSIX',
                              'Operating System :: Unix',
                              'Operating System :: MacOS',
                              'Programming Language :: Python :: 2',
                              'Programming Language :: Python :: 2.7',
                              'Programming Language :: Python :: 3',
                              'Programming Language :: Python :: 3.4',
                              'Programming Language :: Python :: 3.5',
                              'Programming Language :: Python :: 3.6'])
