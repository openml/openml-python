import os
import setuptools
import sys

with open("openml/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
requirements = []
dependency_links = []
with open(requirements_file) as fh:
    for line in fh:
        line = line.strip()
        if line:
            # Make sure the github URLs work here as well
            split = line.split('@')
            split = split[0]
            split = split.split('/')
            url = '/'.join(split[:-1])
            requirement = split[-1]
            requirements.append(requirement)

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
                 license="GPLv3",
                 url="http://openml.org/",
                 version=version,
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 install_requires=requirements,
                 test_suite="nose.collector",
                 classifiers=['Intended Audience :: Science/Research',
                              'Intended Audience :: Developers',
                              'License :: GPLv3',
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
