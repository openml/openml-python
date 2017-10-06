set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cwd=`pwd`
test_dir=$cwd/tests

cd $TEST_DIR

if [[ "$EXAMPLES" == "true" ]]; then
    nosetests -sv $test_dir/test_examples/
elif [[ "$COVERAGE" == "true" ]]; then
    nosetests --processes=4 --process-timeout=600 -sv --ignore-files="test_OpenMLDemo\.py" --with-coverage --cover-package=$MODULE $test_dir
else
    nosetests --processes=4 --process-timeout=600 -sv --ignore-files="test_OpenMLDemo\.py" $test_dir
fi
