# License: BSD 3-Clause

set -e

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
cd ..
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip
source activate testenv

if [[ -v SCIPY_VERSION ]]; then
    conda install --yes scipy=$SCIPY_VERSION
fi
python --version

if [[ "$TEST_DIST" == "true" ]]; then
    pip install twine nbconvert jupyter_client matplotlib pyarrow pytest pytest-xdist pytest-timeout \
        nbformat oslo.concurrency flaky
    python setup.py sdist
    # Find file which was modified last as done in https://stackoverflow.com/a/4561987
    dist=`find dist -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
    echo "Installing $dist"
    pip install "$dist"
    twine check "$dist"
else
    pip install -e '.[test]'
fi

python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [[ "$DOCPUSH" == "true" ]]; then
    conda install --yes gxx_linux-64 gcc_linux-64 swig
    pip install -e '.[examples,examples_unix]'
fi
if [[ "$COVERAGE" == "true" ]]; then
    pip install codecov pytest-cov
fi
if [[ "$RUN_FLAKE8" == "true" ]]; then
    pip install pre-commit
    pre-commit install
fi

# Install scikit-learn last to make sure the openml package installation works
# from a clean environment without scikit-learn.
pip install scikit-learn==$SKLEARN_VERSION

conda list
