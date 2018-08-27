set -o pipefail

# install documentation building dependencies
pip install --upgrade matplotlib seaborn setuptools nose coverage sphinx pillow sphinx-gallery sphinx_bootstrap_theme cython numpydoc nbformat nbconvert

# delete any previous documentation folder for the branch
# if it exists
if [ -d doc/$1 ]; then
    rm -rf doc/$1
fi


# create the documentation
cd doc && make html 2>&1 | tee ~/log.txt

# create directory with branch name
mkdir $1

# copy content
cp -r build/html/* $1
