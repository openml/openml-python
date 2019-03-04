#!/bin/bash

flake8 --ignore E402,W503 --show-source --max-line-length 100 $options
