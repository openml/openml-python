set -e

run_tests() {
    # Get into a temp directory to run test from the installed scikit learn and
    # check if we  do not leave artifacts
    mkdir -p $TEST_DIR

    cwd=`pwd`
    test_dir=$cwd/tests
    doctest_dir=$cwd/doc

    cd $TEST_DIR
    if [[ "$EXAMPLES" == "true" ]]; then
        pytest -sv $test_dir/test_examples/
    elif [[ "$DOCTEST" == "true" ]]; then
        python -m doctest $doctest_dir/usage.rst
    fi

    if [[ "$COVERAGE" == "true" ]]; then
        PYTEST_ARGS='--cov=openml'
    else
        PYTEST_ARGS=''
    fi

    pytest -n 4 --duration=20 --timeout=600 --timeout-method=thread -sv --ignore='test_OpenMLDemo.py' $PYTEST_ARGS $test_dir
}

if [[ "$RUN_FLAKE8" == "true" ]]; then
    source ci_scripts/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi
