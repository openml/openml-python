#!/bin/bash

# Update /CONTRIBUTING.md if these commands change.
# The reason for not advocating using this script directly is that it
# might not work out of the box on Windows.
flake8 --ignore E402,W503 --show-source --max-line-length 100 $options
mypy openml --ignore-missing-imports --follow-imports skip
