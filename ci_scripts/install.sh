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
pip install pytest pytest-xdist pytest-timeout numpy scipy cython scikit-learn==$SKLEARN_VERSION \
    oslo.concurrency

if [[ "$EXAMPLES" == "true" ]]; then
    pip install matplotlib jupyter notebook nbconvert nbformat jupyter_client \
        ipython ipykernel pandas seaborn
fi
if [[ "$DOCTEST" == "true" ]]; then
    pip install pandas sphinx_bootstrap_theme
fi
if [[ "$COVERAGE" == "true" ]]; then
    pip install codecov pytest-cov
fi
if [[ "$RUN_FLAKE8" == "true" ]]; then
    pip install flake8
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
pip install -e '.[test]'
