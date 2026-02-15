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
issues which are suitable for beginners.  Resolving these issues allows you to start
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
   1. Fork and clone the `openml-python` repository. Then, create a new branch from the ``develop`` branch. If you are new to `git`, see our [detailed documentation](#basic-git-workflow), or rely on your favorite IDE.   
   2. [Install the local dependencies](#install-local-dependencies) to run the tests for your contribution.
   3. [Test your installation](#testing-your-installation) to ensure everything is set up correctly.
4. Implement your contribution. If contributing to the documentation, see [here](#contributing-to-the-documentation).
5. [Create a pull request](#pull-request-checklist). 

### Install Local Dependencies

We recommend following the instructions below to install all requirements locally.
However, it is also possible to use the [openml-python docker image](https://github.com/openml/openml-python/blob/main/docker/readme.md) for testing and building documentation. Moreover, feel free to use any alternative package managers, such as `pip`.


1. To ensure a smooth development experience, we recommend using the `uv` package manager. Thus, first install `uv`. If any Python version already exists on your system, follow the steps below, otherwise see [here](https://docs.astral.sh/uv/getting-started/installation/). 
    ```bash
    pip install uv
    ```
2. Create a virtual environment using `uv` and activate it. This will ensure that the dependencies for `openml-python` do not interfere with other Python projects on your system. 
   ```bash
   uv venv --seed --python 3.8 ~/.venvs/openml-python
   source ~/.venvs/openml-python/bin/activate
   pip install uv # Install uv within the virtual environment
   ```
3. Then install openml with its test dependencies by running
   ```bash
   uv pip install -e .[test]
   ```
   from the repository folder.
   Then configure the pre-commit to be able to run unit tests, as well as [pre-commit](#pre-commit-details) through:
   ```bash
   pre-commit install
   ```

### Testing (Your Installation)
To test your installation and run the tests for the first time, run the following from the repository folder:
```bash
pytest tests
```
For Windows systems, you may need to add `pytest` to PATH before executing the command.

Executing a specific unit test can be done by specifying the module, test case, and test.
You may then run a specific module, test case, or unit test respectively:
```bash
pytest tests/test_datasets/test_dataset.py
pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest
pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest::test_get_data
```

To test your new contribution, add [unit tests](https://github.com/openml/openml-python/tree/develop/tests), and, if needed, [examples](https://github.com/openml/openml-python/tree/develop/examples) for any new functionality being introduced. Some notes on unit tests and examples:
* If a unit test contains an upload to the test server, please ensure that it is followed by a file collection for deletion, to prevent the test server from bulking up. For example, `TestBase._mark_entity_for_removal('data', dataset.dataset_id)`, `TestBase._mark_entity_for_removal('flow', (flow.flow_id, flow.name))`.
* Please ensure that the example is run on the test server by beginning with the call to `openml.config.start_using_configuration_for_example()`, which is done by default for tests derived from `TestBase`.
* Add the `@pytest.mark.sklearn` marker to your unit tests if they have a dependency on scikit-learn.

#### Running Tests That Require Admin Privileges

Some tests require admin privileges on the test server and will be automatically skipped unless you provide an admin API key. For regular contributors, the tests will skip gracefully. For core contributors who need to run these tests locally, you can set up the key by exporting the variable as below before running the tests:

```bash
# For windows
$env:OPENML_TEST_SERVER_ADMIN_KEY = "admin-key"
# For linux/mac
export OPENML_TEST_SERVER_ADMIN_KEY="admin-key"
```

### Pull Request Checklist

You can go to the `openml-python` GitHub repository to create the pull request by [comparing the branch](https://github.com/openml/openml-python/compare) from your fork with the `develop` branch of the `openml-python` repository. When creating a pull request, make sure to follow the comments and structured provided by the template on GitHub.

**An incomplete contribution** -- where you expect to do more work before
receiving a full review -- should be submitted as a `draft`. These may be useful
to: indicate you are working on something to avoid duplicated work,
request broad review of functionality or API, or seek collaborators.
Drafts often benefit from the inclusion of a
[task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
in the PR description.

--- 

# Appendix

## Basic `git` Workflow

The preferred workflow for contributing to openml-python is to
fork the [main repository](https://github.com/openml/openml-python) on
GitHub, clone, check out the branch `develop`, and develop on a new branch
branch. Steps:

0. Make sure you have git installed, and a GitHub account.

1. Fork the [project repository](https://github.com/openml/openml-python)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the openml-python repo from your GitHub account to your
local disk:

   ```bash
   git clone git@github.com:YourLogin/openml-python.git
   cd openml-python
   ```

3. Switch to the ``develop`` branch:

   ```bash
   git checkout develop
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   git checkout -b feature/my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``main`` or ``develop`` branch! 
   To make the nature of your pull request easily visible, please prepend the name of the branch with the type of changes you want to merge, such as ``feature`` if it contains a new feature, ``fix`` for a bugfix, ``doc`` for documentation and ``maint`` for other maintenance on the package.

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   git add modified_files
   git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)


## Pre-commit Details
[Pre-commit](https://pre-commit.com/) is used for various style checking and code formatting.
Before each commit, it will automatically run:
 - [ruff](https://docs.astral.sh/ruff/) a code formatter and linter.
   This will automatically format your code.
   Make sure to take a second look after any formatting takes place,
   if the resulting code is very bloated, consider a (small) refactor.
 - [mypy](https://mypy.readthedocs.io/en/stable/) a static type checker.
   In particular, make sure each function you work on has type hints.
    
If you want to run the pre-commit tests without doing a commit, run:
```bash
$ make check
```
or on a system without make, like Windows:
```bash
$ pre-commit run --all-files
```
Make sure to do this at least once before your first commit to check your setup works.

## Contributing to the Documentation

We welcome all forms of documentation contributions — whether it's Markdown docstrings, tutorials, guides, or general improvements.

Our documentation is written either in Markdown or as a jupyter notebook and lives in the docs/ and examples/ directories of the source code repository.

To preview the documentation locally, you will need to install a few additional dependencies:
```bash
uv pip install -e .[examples,docs]
```
When dependencies are installed, run
```bash
mkdocs serve
```
This will open a preview of the website.