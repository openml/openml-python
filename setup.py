# -*- coding: utf-8 -*-

# License: BSD 3-Clause

import os
import setuptools
import sys

with open("openml/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version {}.{}.{} found. OpenML requires Python 3.5 or higher.'
        .format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

with open(os.path.join("README.md")) as fid:
    README = fid.read()

setuptools.setup(name="openml",
                 author="Matthias Feurer, Jan van Rijn, Arlind Kadra, Pieter Gijsbers, "
                        "Neeratyoy Mallik, Sahithya Ravi, Andreas Müller, Joaquin Vanschoren "
                        "and Frank Hutter",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 maintainer="Matthias Feurer",
                 maintainer_email="feurerm@informatik.uni-freiburg.de",
                 description="Python API for OpenML",
                 long_description=README,
                 long_description_content_type='text/markdown',
                 license="BSD 3-clause",
                 url="http://openml.org/",
                 project_urls={
                     "Documentation": "https://openml.github.io/openml-python/",
                     "Source Code": "https://github.com/openml/openml-python"
                 },
                 version=version,
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 python_requires=">=3.5",
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
                         'oslo.concurrency',
                         'flaky',
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
                     ],
                     'examples_unix': [
                         'fanova',
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
                              'Programming Language :: Python :: 3.5',
                              'Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7'])
