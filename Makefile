# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean: 
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s -v openml
test-doc:
	$(NOSETESTS) -s -v doc/*.rst

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage openml

test: test-code test-sphinxext test-doc
