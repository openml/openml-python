import os
import setuptools


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
            # Add the rest of the URL to the dependency links to allow
            # setup.py test to work
            if 'git+https' in url:
                dependency_links.append(line.replace('git+', ''))


setuptools.setup(name="openml",
                 author="Matthias Feurer",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 maintainer="Matthias Feurer",
                 maintainer_email="feurerm@informatik.uni-freiburg.de",
                 description="Python API for OpenML",
                 license="GPLv3",
                 url="http://openml.org/",
                 version="0.2.1",
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 install_requires=[],
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
                 ],
                 dependency_links=[
                     "http://github.com/mfeurer/liac-arff/archive/master.zip"
                     "#egg=liac-arff-2.1.1dev"])