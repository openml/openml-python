# OpenML Python Container

This docker container has the latest version of openml-python downloaded and pre-installed.
It can also be used by developers to run unit tests or build the docs in 
a fresh and/or isolated unix environment. 
This document contains information about:

 1. [Usage](#usage): how to use the image and its main modes.
 2. [Using local or remote code](#using-local-or-remote-code): useful when testing your own latest changes.
 3. [Versions](#versions): identify which image to use.
 4. [Development](#for-developers): information about the Docker image for developers.

*note:* each docker image is shipped with a readme, which you can read with:
`docker run --entrypoint=/bin/cat openml/openml-python:TAG readme.md`

## Usage

There are three main ways to use the image: running a pre-installed Python environment,
running tests, and building documentation.

### Running `Python` with pre-installed `OpenML-Python` (default):

To run `Python` with a pre-installed `OpenML-Python` environment run:

```text
docker run -it openml/openml-python
```

this accepts the normal `Python` arguments, e.g.:

```text
docker run openml/openml-python -c "import openml; print(openml.__version__)"
```

if you want to run a local script, it needs to be mounted first. Mount it into the
`openml` folder:

```
docker run -v PATH/TO/FILE:/openml/MY_SCRIPT.py openml/openml-python MY_SCRIPT.py
```

### Running unit tests

You can run the unit tests by passing `test` as the first argument.
It also requires a local or remote repository to be specified, which is explained 
[below]((#using-local-or-remote-code). For this example, we specify to test the
`develop` branch:

```text
docker run openml/openml-python test develop
```

### Building documentation

You can build the documentation by passing `doc` as the first argument, 
you should [mount]((https://docs.docker.com/storage/bind-mounts/#start-a-container-with-a-bind-mount)) 
an output directory in which the docs will be stored. You also need to provide a remote
or local repository as explained in [the section below]((#using-local-or-remote-code).
In this example, we build documentation for the `develop` branch.
On Windows:

```text
    docker run --mount type=bind,source="E:\\files/output",destination="/output" openml/openml-python doc develop
```

on Linux:
```text
    docker run --mount type=bind,source="./output",destination="/output" openml/openml-python doc develop
```
    
see [the section below]((#using-local-or-remote-code) for running against local changes
or a remote branch.

*Note: you can forgo mounting an output directory to test if the docs build successfully,
but the result will only be available within the docker container under `/openml/docs/build`.*

## Using local or remote code

You can build docs or run tests against your local repository or a Github repository.
In the examples below, change the `source` to match the location of your local repository.

### Using a local repository

To use a local directory, mount it in the `/code` directory,  on Windows:

```text
    docker run --mount type=bind,source="E:\\repositories/openml-python",destination="/code" openml/openml-python test
```

on Linux:
```text
    docker run --mount type=bind,source="/Users/pietergijsbers/repositories/openml-python",destination="/code" openml/openml-python test
```

when building docs, you also need to mount an output directory as shown above, so add both:

```text
docker run --mount type=bind,source="./output",destination="/output" --mount type=bind,source="/Users/pietergijsbers/repositories/openml-python",destination="/code" openml/openml-python doc
```

### Using a Github repository
Building from a remote repository requires you to specify a branch.
The branch may be specified by name directly if it exists on the original repository (https://github.com/openml/openml-python/):

    docker run --mount type=bind,source=PATH_TO_OUTPUT,destination=/output openml/openml-python [test,doc] BRANCH

Where `BRANCH` is the name of the branch for which to generate the documentation.
It is also possible to build the documentation from the branch on a fork,
in this case the `BRANCH` should be specified as `GITHUB_NAME#BRANCH` (e.g. 
`PGijsbers#my_feature_branch`) and the name of the forked repository should be `openml-python`.

## For developers
This section contains some notes about the structure of the image, 
intended for those who want to work on it.

### Added Directories
The `openml/openml-python` image is built on a vanilla `python:3` image.
Additionally, it contains the following files are directories:

 - `/openml`: contains the openml-python repository in the state with which the image 
   was built by default. If working with a `BRANCH`, this repository will be set to 
   the `HEAD` of `BRANCH`.
 - `/openml/venv/`: contains the used virtual environment for `doc` and `test`. It has
   `openml-python` dependencies pre-installed.  When invoked with `doc` or `test`, the 
   dependencies will be updated based on the `setup.py` of the `BRANCH` or mounted `/code`.
 - `/scripts/startup.sh`: the entrypoint of the image. Takes care of the automated features (e.g. `doc` and `test`).

## Building the image
To build the image yourself, execute `docker build -f Dockerfile .` from the `docker`
directory of the `openml-python` repository. It will use the `startup.sh` as is, so any 
local changes will be present in the image.
