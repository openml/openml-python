#!/bin/bash

flake8 --ignore E402,W503 --show-source --max-line-length 100 $options
mypy openml --ignore-missing-imports --follow-imports skip
