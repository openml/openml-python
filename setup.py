# -*- coding: utf-8 -*-

import setuptools
import sys

with open("openml/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

# Using Python setup.py install will try to build numpy which is prone to failure and
# very time consuming anyway.
if len(sys.argv) > 1 and sys.argv[1] == 'install':
    print('Please install this package with pip: `pip install -e .` '
          'Installation requires pip>=10.0.')
    sys.exit(1)

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version {}.{}.{} found. OpenML requires Python 3.5 or higher.'
        .format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

setuptools.setup(name="openml",
                 author="Matthias Feurer, Jan van Rijn, Arlind Kadra, Andreas Müller, "
                        "Pieter Gijsbers and Joaquin Vanschoren",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 maintainer="Matthias Feurer",
                 maintainer_email="feurerm@informatik.uni-freiburg.de",
                 description="Python API for OpenML",
                 license="BSD 3-clause",
                 url="http://openml.org/",
                 project_urls={
                     "Documentation": "https://openml.github.io/openml-python/",
                     "Source Code": "https://github.com/openml/openml-python"
                 },
                 version=version,
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 install_requires=[
                     'liac-arff>=2.4.0',
                     'xmltodict',
                     'requests',
                     'scikit-learn>=0.18',
                     'python-dateutil',  # Installed through pandas anyway.
                     'pandas>=0.19.2',
                     'scipy>=0.13.3',
                     'numpy>=1.6.2'
                 ],
                 extras_require={
                     'test': [
                         'nbconvert',
                         'jupyter_client',
                         'matplotlib',
                         'pytest',
                         'pytest-xdist',
                         'pytest-timeout',
                         'nbformat',
                         'oslo.concurrency'
                     ],
                     'examples': [
                         'matplotlib',
                         'jupyter',
                         'notebook',
                         'nbconvert',
                         'nbformat',
                         'jupyter_client',
                         'ipython',
                         'ipykernel',
                         'seaborn'
                     ]
                 },
                 test_suite="pytest",
                 classifiers=['Intended Audience :: Science/Research',
                              'Intended Audience :: Developers',
                              'License :: OSI Approved :: BSD License',
                              'Programming Language :: Python',
                              'Topic :: Software Development',
                              'Topic :: Scientific/Engineering',
                              'Operating System :: POSIX',
                              'Operating System :: Unix',
                              'Operating System :: MacOS',
                              'Programming Language :: Python :: 3',
                              'Programming Language :: Python :: 3.4',
                              'Programming Language :: Python :: 3.5',
                              'Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7'])
