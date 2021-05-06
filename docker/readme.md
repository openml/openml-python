# OpenML Python Container

This docker container has the latest development version of openml-python downloaded and pre-installed.
It can be used to run the unit tests or build the docs in a fresh and/or isolated unix environment.
Instructions only tested on a Windows host machine.

First pull the docker image:

    docker pull openml/openml-python

## Usage
`docker run -it openml/openml-python [-|DOC|TEST] [BRANCH]`

The image is designed to work with two specified directories which may be mounted ([`docker --mount documentation`](https://docs.docker.com/storage/bind-mounts/#start-a-container-with-a-bind-mount)).
You can mount your openml-python folder to the `/code` directory to run tests or build docs on your local files.
You can mount an `/output` directory to which the container will write output (currently only used for docs).
Each can be mounted by adding a `--mount type=bind,source=SOURCE,destination=/DESTINATION` where `SOURCE` is the absolute path to your code or output directory, and `DESTINATION` is either `code` or `output`.
  
E.g. mounting a code directory: 

    docker run -i --mount type=bind,source="E:\\repositories/openml-python",destination="/code" -t openml/openml-python

E.g. mounting an output directory: 

    docker run -i --mount type=bind,source="E:\\files/output",destination="/output" -t openml/openml-python

You can mount both at the same time.

### Bash (default)
By default bash is invoked, you should also use the `-i` flag when starting the container so it processes input: 

    docker run -it openml/openml-python

### Building Documentation
There are two ways to build documentation, either directly from the `HEAD` of a branch on Github or from your local directory.

#### Building from a local repository
Building from a local directory requires you to mount it to the ``/code`` directory:

    docker run --mount type=bind,source=PATH_TO_REPOSITORY,destination=/code -t openml/openml-python doc

The produced documentation will be in your repository's ``doc/build`` folder.
If an `/output` folder is mounted, the documentation will *also* be copied there.

#### Building from an online repository
Building from a remote repository requires you to specify a branch.
The branch may be specified by name directly if it exists on the original repository (https://github.com/openml/openml-python/):

    docker run --mount type=bind,source=PATH_TO_OUTPUT,destination=/output -t openml/openml-python doc BRANCH

Where `BRANCH` is the name of the branch for which to generate the documentation.
It is also possible to build the documentation from the branch on a fork, in this case the `BRANCH` should be specified as `GITHUB_NAME#BRANCH` (e.g. `PGijsbers#my_feature`) and the name of the forked repository should be `openml-python`.

### Running tests
There are two ways to run tests, either directly from the `HEAD` of a branch on Github or from your local directory.
It works similar to building docs, but should specify `test` as mode.
For example, to run tests on your local repository:

    docker run --mount type=bind,source=PATH_TO_REPOSITORY,destination=/code -t openml/openml-python test
    
## Troubleshooting

When you are mounting a directory you can check that it is mounted correctly by running the image in bash mode.
Navigate to the `/code` and `/output` directories and see if the expected files are there.
If e.g. there is no code in your mounted `/code`, you should double-check the provided path to your host directory.

## Notes for developers
This section contains some notes about the structure of the image, intended for those who want to work on it.

### Added Directories
There is a `/omlp` directory which by default contains the openml-python repository in the state with which the image was built.
The used virtual environment resides in the `/omlp/venv/` folder.
The startup script is located in `/scripts/startup.sh`.
Otherwise it is built on a vanilla `python:3` image.

## Building the image
To build the image yourself, execute `docker build -f Dockerfile .` from this directory.
It will use the `startup.sh` as is, so any local changes will be present in the image.
