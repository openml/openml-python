set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cwd=`pwd`
test_dir=$cwd/tests
doctest_dir=$cwd/doc

cd $TEST_DIR

if [[ "$EXAMPLES" == "true" ]]; then
    nosetests -sv $test_dir/test_examples/
elif [[ "$DOCTEST" == "true" ]]; then
    python -m doctest $doctest_dir/usage.rst
elif [[ "$COVERAGE" == "true" ]]; then
    nosetests --processes=4 --process-timeout=600 -sv --ignore-files="test_OpenMLDemo\.py" --with-coverage --cover-package=$MODULE $test_dir
else
    nosetests --processes=4 --process-timeout=600 -sv --ignore-files="test_OpenMLDemo\.py" $test_dir
fi
