# Entry script to allow docker to be ran for bash, tests and docs.
# $1: Mode, optional. If provided, must be 'test' or 'doc'.
# $2: Branch, optional. If provided, execute MODE on branch.
# No mode specified, just use bash:
if [ -z "$1" ]; then
  echo "Executing in BASH mode."
  bash
  exit
fi
if [ -n "$2" ]; then
  # check branch exists
  # if not exists: error, branch does not exist (did you push?).
  # if exists, pull into ``omlp``.
  echo "Branch not implemented yet."
  code_dir="/omlp"
  exit 1
else
  # `code` must be mounted
  if [ -d "/code" ]; then
    code_dir="/code"
  else
    echo "No branch is specified, and no folder mounted to '/code'. " >> /dev/stderr
    exit 1
  fi
fi
source /omlp/venv/bin/activate
cd $code_dir
# The most recent ``master`` is already installed, but we want to update any outdated dependencies
pip install -e .[test,examples,docs,examples_unix]
if [ "$1" == "test" ]; then
  pytest -n 4 --durations=20 --timeout=600 --timeout-method=thread --dist load -sv
fi
if [ "$1" == "doc" ]; then
  cd doc
  make html
  make linkcheck
  if [ -d "/output" ]; then
    cp -r /omlp/doc/build /output
  fi
fi
