# Documentation 

This directory contains all `.rst` files used by `tavis-ci` to generate the documentations. 

### Contributing to the documentation

For building the documentation, you will need sphinx, sphinx-bootstrap-theme, sphinx-gallery and numpydoc.

```
$ pip install sphinx sphinx-bootstrap-theme sphinx-gallery numpydoc
```

When dependencies are installed, run

```
$ sphinx-build -b html doc YOUR_PREFERRED_OUTPUT_DIRECTORY
```

Below is a list of currently available pages or sections that need to be updated:

* Contributing - page: `contributing.rst`
* Front page - page: `index.rst`
* API - Page: papi.rst`
* Progress - page: `progress.rst`
* Usage	- page: `usage.rst` 


