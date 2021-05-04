# OpenML Python Container

This docker container has the latest master version of openml-python downloaded pre-installed.
It can also be used to run the unit tests or build the docs in a fresh and/or isolated unix environment.
Instructions only tested on a Windows host machine.

## Usage
`docker run openml-python [-|DOC|TEST|PYTHON] [BRANCH]`

The image is designed to also work with two specified directories which may be mounted.
You can mount your openml-python folder to the `code` directory, which will be used if available.
You can mount an `output` directory to which the container will write output (e.g. docs).
Each can be mounted by adding a `--mount source=SOURCE,destination=/DESTINATION` where `SOURCE` is the absolute path to your code or output directory, and `DESTINATION` is either `code` or `output`.  
[ ? ] E.g. `docker run --mount source=E:\\repositories\\openml-python,destination=/code openml-python`.

### Bash (default)
By default bash is invoked.

### Running Python
Running a Python console with openml-python already installed: [ ? ] `docker run openml-python python`
It's effectively a shorthand for `docker run --entrypoint /omlp/venv/bin/python openml-python`.

### Building Documentation
There are two ways to build documentation, either directly from the `HEAD` of a branch on Github, or from your local directory.
Either way the first argument 

**From a Github branch:** Requires you to specify the branch name to build the docs for, and to mount a folder which the docs will be saved to.
Format: `docker run openml-python doc BRANCH_NAME --mount source=PATH_TO_LOCAL_DIRECTORY,destination=/output`
Example: [ ? ] `docker run openml-python doc develop --mount source=E:\\tmp\\openml,destination=/output`

**From a local directory:** Useful if you're iterating 


## Added Directories
There is a `/omlp` directory which by default contains the openml-python repository in the state with which the image was built.
The used virtual environment resides in the `/omlp/venv/` folder.
The startup script is located in `/scripts/startup.sh`.
Otherwise it is built on a vanilla `python:3` image.

## Building the image
To build the image yourself, execute `docker build -f Dockerfile .` from this directory.
It will use the `startup.sh` as is, so any local changes will be present in the image.