# Entry script to switch between the different Docker functionalities.
# By default, execute Python with OpenML pre-installed
#
# Entry script to allow docker to be ran for bash, tests and docs.
# The script assumes a code repository can be mounted to ``/code`` and an output directory to ``/output``.
# Executes ``mode`` on ``branch`` or the provided ``code`` directory.
# $1: Mode, optional. Options:
#        - test: execute unit tests
#        - doc: build documentation, requires a mounted ``output`` directory if built from a branch.
#        - if not provided: execute bash.
# $2: Branch, optional.
#        Mutually exclusive with mounting a ``code`` directory.
#        Can be a branch on a Github fork, specified with the USERNAME#BRANCH format.
#        The test or doc build is executed on this branch.

if [[ ! ( $1 = "doc" || $1 = "test" ) ]]; then
  cd openml
  source venv/bin/activate
  python "$@"
  exit 0
fi

# doc and test modes require mounted directories and/or specified branches
if ! [ -d "/code" ] && [ -z "$2" ]; then
  echo "To perform $1 a code repository must be mounted to '/code' or a branch must be specified." >> /dev/stderr
  exit 1
fi
if [ -d "/code" ] && [ -n "$2" ]; then
  # We want to avoid switching the git environment from within the docker container
  echo "You can not specify a branch for a mounted code repository." >> /dev/stderr
  exit 1
fi
if [ "$1" == "doc" ]  && [ -n "$2" ] && ! [ -d "/output" ]; then
    echo "To build docs from an online repository, you need to mount an output directory." >> /dev/stderr
    exit 1
fi

if [ -n "$2" ]; then
  # if a branch is provided, we will pull it into the `openml` local repository that was created with the image.
  cd openml
  if [[ $2 == *#* ]]; then
    # If a branch is specified on a fork (with NAME#BRANCH format), we have to construct the url before pulling
    # We add a trailing '#' delimiter so the second element doesn't get the trailing newline from <<<
    readarray -d '#' -t fork_name_and_branch<<<"$2#"
    fork_url="https://github.com/${fork_name_and_branch[0]}/openml-python.git"
    fork_branch="${fork_name_and_branch[1]}"
    echo git fetch "$fork_url" "$fork_branch":branch_from_fork
    git fetch "$fork_url" "$fork_branch":branch_from_fork
    branch=branch_from_fork
  else
    git fetch origin "$2"
    branch=$2
  fi
  if ! git checkout "$branch" ; then
    echo "Could not checkout $branch. If the branch lives on a fork, specify it as USER#BRANCH. Make sure to push the branch." >> /dev/stderr
    exit 1
  fi
  git pull
  code_dir="/openml"
else
  code_dir="/code"
fi

source /openml/venv/bin/activate
cd $code_dir
# The most recent ``main`` is already installed, but we want to update any outdated dependencies
pip install -e .[test,examples,docs,examples_unix]

if [ "$1" == "test" ]; then
  pytest -n 4 --durations=20 --timeout=600 --timeout-method=thread --dist load -sv
fi

if [ "$1" == "doc" ]; then
  cd doc
  make html
  make linkcheck
  if [ -d "/output" ]; then
    cp -r /openml/doc/build /output
  fi
fi