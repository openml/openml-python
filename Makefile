# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

all: clean inplace test

check:
	pre-commit run --all-files

clean:
	$(PYTHON) setup.py clean
	rm -rf dist openml.egg-info

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(PYTEST) -s -v tests
test-doc:
	$(PYTEST) -s -v doc/*.rst

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) -s -v --cov=. tests

test: test-code test-sphinxext test-doc
