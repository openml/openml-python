set -eo pipefail

# set -euo pipefail -> Breaking travis for the moment
# If in the future the bug is fixed, everything can be wrapped in the condition below.
# if ! [[ -z ${DOCPUSH+x} ]]; then
# fi

if [[ "$DOCPUSH" == "true" ]]; then

    # install documentation building dependencies
    pip install --upgrade matplotlib seaborn setuptools nose coverage sphinx pillow sphinx-gallery sphinx_bootstrap_theme cython numpydoc nbformat nbconvert

    # $1 is the branch name
    # $2 is the global variable where we set the script status

    if ! { [ $1 = "master" ] || [ $1 = "develop" ]; }; then
        { echo "fail"; exit 1; }
    fi

    # delete any previous documentation folder
    if [ -d doc/$1 ]; then
        rm -rf doc/$1
    fi

    # create the documentation
    cd doc && make html 2>&1

    # create directory with branch name
    # the documentation for dev/stable from git will be stored here
    mkdir $1

    # get previous documentation from github
    git clone https://github.com/openml/openml-python.git --branch gh-pages --single-branch

    # copy previous documentation
    cp -r openml-python/. $1
    rm -rf openml-python

    # if the documentation for the branch exists, remove it
    if [ -d $1/$1 ]; then
        rm -rf $1/$1
    fi

    # copy the updated documentation for this branch
    mkdir $1/$1
    cp -r build/html/. $1/$1

    function set_return() {
        # $1 is the variable where we save the script outcome
        local __result=$1
        local  status='success'
        eval $__result="'$status'"
    }

    set_return "$2"
fi
