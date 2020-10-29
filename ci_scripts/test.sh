# License: BSD 3-Clause

set -e

# check status and branch before running the unit tests
before="`git status --porcelain -b`"
before="$before"
# storing current working directory
curr_dir=`pwd`

run_tests() {
    # Get into a temp directory to run test from the installed scikit learn and
    # check if we  do not leave artifacts
    mkdir -p $TEST_DIR

    cwd=`pwd`
    test_dir=$cwd/tests

    cd $TEST_DIR

    if [[ "$COVERAGE" == "true" ]]; then
        PYTEST_ARGS='--cov=openml --long'
    else
        PYTEST_ARGS=''
    fi

    pytest -n 4 --durations=20 --timeout=600 --timeout-method=thread -sv $PYTEST_ARGS $test_dir
}

if [[ "$RUN_FLAKE8" == "true" ]]; then
    pre-commit run --all-files
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi

# changing directory to stored working directory
cd $curr_dir
# check status and branch after running the unit tests
# compares with $before to check for remaining files
after="`git status --porcelain -b`"
if [[ "$before" != "$after" ]]; then
    echo 'git status from before: '$before
    echo 'git status from after: '$after
    echo "All generated files have not been deleted!"
    exit 1
fi
