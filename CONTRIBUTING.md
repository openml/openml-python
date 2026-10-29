# Contributing to `openml-python`

This document describes the workflow on how to contribute to the openml-python package.
If you are interested in connecting a machine learning package with OpenML (i.e.
write an openml-python extension) or want to find other ways to contribute, see [this page](https://openml.github.io/openml-python/main/contributing.html#contributing).

## Scope of the package

The scope of the OpenML Python package is to provide a Python interface to
the OpenML platform which integrates well with Python's scientific stack, most
notably [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/) and
[pandas](https://pandas.pydata.org/).
To reduce opportunity costs and demonstrate the usage of the package, it also
implements an interface to the most popular machine learning package written
in Python, [scikit-learn](http://scikit-learn.org/stable/index.html).
Thereby it will automatically be compatible with many machine learning
libraries written in Python.

We aim to keep the package as light-weight as possible, and we will try to
keep the number of potential installation dependencies as low as possible.
Therefore, the connection to other machine learning libraries such as
*pytorch*, *keras* or *tensorflow* should not be done directly inside this
package, but in a separate package using the OpenML Python connector.
More information on OpenML Python connectors can be found [here](https://openml.github.io/openml-python/main/contributing.html#contributing).

## Determine what contribution to make

Great! You've decided you want to help out. Now what?

All contributions should be linked to issues on the [GitHub issue tracker](https://github.com/openml/openml-python/issues).
In particular for new contributors, the *good first issue* label should help you find
issues which are suitable for beginners. Resolving these issues allows you to start
contributing to the project without much prior knowledge. Your assistance in this area
will be greatly appreciated by the more experienced developers as it helps free up
their time to concentrate on other issues.

If you encounter a particular part of the documentation or code that you want to improve,
but there is no related open issue yet, open one first.
This is important since you can first get feedback or pointers from experienced contributors.

To let everyone know you are working on an issue, please leave a comment that states you will work on the issue
(or, if you have the permission, *assign* yourself to the issue). This avoids double work!

## Contributing Workflow Overview

To contribute to the openml-python package, follow these steps:

0. Determine how you want to contribute (see above).
1. Set up your local development environment.
   1. Fork and clone the `openml-python` repository. Then, create a new branch from the ``main`` branch. If you are new to `git`, see our [detailed documentation](#basic-git-workflow), or rely on your favorite IDE.
   2. [Install the local dependencies](#install-local-dependencies) to run the tests for your contribution.
   3. [Test your installation](#testing-your-installation) to ensure everything is set up correctly.
2. Implement your contribution. If contributing to the documentation, see [here](#contributing-to-the-documentation).
3. [Create a pull request](#pull-request-checklist).

### Install Local Dependencies

We recommend following the instructions below to install the requirements locally.
It is also possible to use the [openml-python Docker image](https://github.com/openml/openml-python/blob/main/docker/readme.md) for testing and building documentation, or the entire OpenML service stack using docker as described [here](https://github.com/openml/services).

These instructions use `uv`; if you choose another package manager, adapt the commands to your environment.

1. **Install `uv`.** Follow the [official `uv` installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.

2. **Install Python 3.12.** The project currently supports Python `>=3.10,<3.15`; the examples below use Python 3.12. Install it with:

   ```bash
   uv python install 3.12
   ```

3. **Create a virtual environment.** From the root of the repository (the directory containing `pyproject.toml`), run:

   ```bash
   uv venv --python 3.12
   ```

   This creates a `.venv` directory inside the repository.

4. **Activate the virtual environment.** Use the command for your operating system and shell.

   **Linux and macOS — Bash or zsh**

   ```bash
   source .venv/bin/activate
   ```

   **Windows — PowerShell**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Windows — Command Prompt**

   ```bat
   .venv\Scripts\activate.bat
   ```

5. **Install openml with its test dependencies.** From the repository folder, run:

   ```bash
   uv pip install -e ".[test]"
   ```

   The editable installation means changes you make to the source code are available without reinstalling the package after every edit.

6. **Install the pre-commit hooks.** These run the project's code-quality checks before commits:

   ```bash
   pre-commit install
   ```

7. **Verify the installation.** Check that Python imports the package from your local checkout:

   ```bash
   python -c "import openml; print(openml.__file__)"
   ```

   The output should point to the `openml` package in your repository. To check which Python interpreter is active, run `python -c "import sys; print(sys.executable)"`.

### Testing (Your Installation)

> We are migrating our unit tests from unittest to pytest so test classes are being removed from the code base.

The test suite is an important part of contributing to `openml-python`, but **the complete suite can take a substantial amount of time to run**, especially when tests interact with OpenML services. During development, start with the smallest relevant test and broaden the selection as your change stabilizes.

Tests are marked according to the server they use:

- `test_server` marks tests that normally require the OpenML test server (https://test.openml.org). When the local OpenML services stack is running and `OPENML_USE_LOCAL_SERVICES=true` is set, these tests can use the local services instead.
- `production_server` marks tests that interact with the production OpenML server (https://www.openml.org). Setting `OPENML_USE_LOCAL_SERVICES=true` does **not** redirect these tests, so exclude them from normal local development runs.

#### Run a focused test

Run the test file or individual test that covers your change. For example:

```bash
python -m pytest tests/test_datasets/test_dataset.py::test__check_qualities
```

Replace the example with the test relevant to your contribution. Check the test's markers and fixtures to determine whether it needs an OpenML server.

#### Run tests that do not need a server

When possible, prefer tests that run entirely locally. To exclude tests marked for either the production or test server, use:

```bash
python -m pytest tests -m "not production_server and not test_server"
```

This is a useful starting point for local feedback, but it does not replace tests for functionality that depends on the API. Add or update tests for your change and run the relevant ones even when they need a server.
While existing tests may continue to connect to the production server, new tests should avoid doing so unless explicitly approved by a maintainer. New tests may connect to the test server if there is a clear need for it, but it is best avoided otherwise.

#### Run `test_server` tests against local OpenML services

It is possible to have tests connect to locally running services.
The local-services configuration applies to tests marked `test_server` only. Tests marked `production_server` explicitly target production and are not redirected. 
Please follow the instructions from the [`openml/services` repository](https://github.com/openml/services) on how to start local services. For the openml-python tests, the provided "rest-api" should be used. 

Set `OPENML_USE_LOCAL_SERVICES` in the same terminal session in which you run the pytests.

**Linux and macOS — Bash or zsh**

```bash
export OPENML_USE_LOCAL_SERVICES=true
python -m pytest tests/test_datasets/test_dataset.py -sv -m "test_server"
```

**Windows — PowerShell**

```powershell
$env:OPENML_USE_LOCAL_SERVICES = "true"
python -m pytest tests/test_datasets/test_dataset.py -sv -m "test_server"
```

Replace the example test file with the one relevant to your change. To run all tests except those targeting production while using local services for tests marked `test_server`, use `-m "not production_server"` instead of `-m "test_server"`.

It is not preferable run production-server tests as part of this local workflow often. If a test fails, first check its marker, whether local-services mode is enabled in the same terminal, and whether the local services are healthy before treating the failure as a problem with your code.

#### Running Tests That Require Admin Privileges

Some tests require admin privileges on the test server and will be automatically skipped unless you provide an admin API key. For regular contributors, the tests will skip gracefully. For core contributors who need to run these tests locally, set the variable before running the tests:

**Linux and macOS — Bash or zsh**

```bash
export OPENML_TEST_SERVER_ADMIN_KEY="admin-key"
```

**Windows — PowerShell**

```powershell
$env:OPENML_TEST_SERVER_ADMIN_KEY = "admin-key"
```


#### Diagnosing Slow Tests

If you suspect a test (or the suite as a whole) is running too slowly, `pytest` exposes several options that are useful for investigating test runtimes:

```bash
# Show the 20 slowest tests (use 0 to list every test's duration)
python -m pytest tests --durations=20

# Fail any test that exceeds the given timeout (requires pytest-timeout)
python -m pytest tests --timeout=600

# Investigate fixture/setup costs without actually running the tests
python -m pytest tests --setup-only

# Profile a specific module, class, or test
python -m pytest tests/test_datasets/test_dataset.py --durations=0

# Profile tests without production- or test-server interactions
python -m pytest tests --durations=0 -m "not production_server and not test_server"

# Run tests in parallel against local services (requires pytest-xdist; set OPENML_USE_LOCAL_SERVICES=true first)
python -m pytest tests -n 4 --dist=load --durations=0 -m "not production_server"
```

Combining these options with marker filters (`production_server`, `test_server`, and `sklearn`) makes it straightforward to narrow the investigation down to the slow tests without changing project configuration.

### Pull Request Checklist

You can go to the `openml-python` GitHub repository to create the pull request by [comparing the branch](https://github.com/openml/openml-python/compare) from your fork with the `main` branch of the `openml-python` repository. When creating a pull request, make sure to follow the comments and structure provided by the template on GitHub.

**An incomplete contribution** — where you expect to do more work before receiving a full review — should be submitted as a `draft`. These may be useful to indicate you are working on something to avoid duplicated work, request broad review of functionality or API, or seek collaborators. Drafts often benefit from a task list in the PR description.

---

# Appendix

## Basic `git` Workflow

The preferred workflow for contributing to openml-python is to fork the [main repository](https://github.com/openml/openml-python) on GitHub, clone, and develop on a new branch:

0. Make sure you have git installed, and a GitHub account.
1. Fork the [project repository](https://github.com/openml/openml-python) by clicking on the 'Fork' button near the top right of the page. This creates a copy of the code under your GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the openml-python repo from your GitHub account to your local disk:

   ```bash
   git clone git@github.com:YourLogin/openml-python.git
   cd openml-python
   ```

3. Create a `feature` branch to hold your development changes:

   ```bash
   git switch -c feature/my-feature
   ```

   Always use a `feature` branch. It's good practice to never work on the `main` branch! To make the nature of your pull request easily visible, please prepend the name of the branch with the type of changes you want to merge, such as `feature` if it contains a new feature, `fix` for a bugfix, `doc` for documentation and `maint` for other maintenance on the package.

4. Develop the feature on your feature branch. Add changed files using `git add` and then `git commit` files:

   ```bash
   git add modified_files
   git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   git push -u origin feature/my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork) to create a pull request from your fork.

(If any of the above seems like magic to you, please look up the [Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

## Pre-commit Details

[Pre-commit](https://pre-commit.com/) is used for various style checking and code formatting.
Before each commit, it will automatically run:

- [ruff](https://docs.astral.sh/ruff/) a code formatter and linter. This will automatically format your code. Make sure to take a second look after any formatting takes place; if the resulting code is very bloated, consider a (small) refactor.
- [mypy](https://mypy.readthedocs.io/en/stable/) a static type checker. In particular, make sure each function you work on has type hints.

If you want to run the pre-commit tests without doing a commit, run:

```bash
make check
```

or on a system without `make`, like Windows:

```bash
pre-commit run --all-files
```

Make sure to do this at least once before your first commit to check your setup works.

## Contributing to the Documentation

We welcome all forms of documentation contributions — whether it's Markdown docstrings, tutorials, guides, or general improvements.

Our documentation is written either in Markdown or as a jupyter notebook and lives in the `docs/` and `examples/` directories of the source code repository.

To preview the documentation locally, you will need to install a few additional dependencies:

```bash
uv pip install -e ".[examples,docs]"
```

When dependencies are installed, run:

```bash
mkdocs serve
```

This will open a preview of the website.
