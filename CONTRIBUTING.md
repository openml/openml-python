This document describes the workflow on how to contribute to the openml-python package.
If you are interested in connecting a machine learning package with OpenML (i.e.
write an openml-python extension) or want to find other ways to contribute, see [this page](https://openml.github.io/openml-python/main/contributing.html#contributing).

Scope of the package
--------------------

The scope of the OpenML Python package is to provide a Python interface to
the OpenML platform which integrates well with Python's scientific stack, most
notably [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/) and
[pandas](https://pandas.pydata.org/).
To reduce opportunity costs and demonstrate the usage of the package, it also
implements an interface to the most popular machine learning package written
in Python, [scikit-learn](http://scikit-learn.org/stable/index.html).
Thereby it will automatically be compatible with many machine learning
libraries written in Python.

We aim to keep the package as light-weight as possible and we will try to
keep the number of potential installation dependencies as low as possible.
Therefore, the connection to other machine learning libraries such as
*pytorch*, *keras* or *tensorflow* should not be done directly inside this
package, but in a separate package using the OpenML Python connector.
More information on OpenML Python connectors can be found [here](https://openml.github.io/openml-python/main/contributing.html#contributing).

Reporting bugs
--------------
We use GitHub issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/openml/openml-python/issues)
   or [pull requests](https://github.com/openml/openml-python/pulls).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, openml, scikit-learn, numpy, and scipy versions. This information
   can be found by running the following code snippet:
```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import openml; print("OpenML", openml.__version__)
```

Determine what contribution to make
-----------------------------------
Great! You've decided you want to help out. Now what?
All contributions should be linked to issues on the [Github issue tracker](https://github.com/openml/openml-python/issues).
In particular for new contributors, the *good first issue* label should help you find
issues which are suitable for beginners.  Resolving these issues allow you to start
contributing to the project without much prior knowledge. Your assistance in this area 
will be greatly appreciated by the more experienced developers as it helps free up 
their time to concentrate on other issues.

If you encountered a particular part of the documentation or code that you want to improve,
but there is no related open issue yet, open one first.
This is important since you can first get feedback or pointers from experienced contributors.

To let everyone know you are working on an issue, please leave a comment that states you will work on the issue
(or, if you have the permission, *assign* yourself to the issue). This avoids double work!

General git workflow
--------------------

The preferred workflow for contributing to openml-python is to
fork the [main repository](https://github.com/openml/openml-python) on
GitHub, clone, check out the branch `develop`, and develop on a new branch
branch. Steps:

1. Fork the [project repository](https://github.com/openml/openml-python)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the openml-python repo from your GitHub account to your
local disk:

   ```bash
   $ git clone git@github.com:YourLogin/openml-python.git
   $ cd openml-python
   ```

3. Switch to the ``develop`` branch:

   ```bash
   $ git checkout develop
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b feature/my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``main`` or ``develop`` branch! 
   To make the nature of your pull request easily visible, please prepend the name of the branch with the type of changes you want to merge, such as ``feature`` if it contains a new feature, ``fix`` for a bugfix, ``doc`` for documentation and ``maint`` for other maintenance on the package.

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the
   [pep8 style guide](https://www.python.org/dev/peps/pep-0008/).
   With the following exceptions or additions:
    - The max line length is 100 characters instead of 80.
    - When creating a multi-line expression with binary operators, break before the operator.
    - Add type hints to all function signatures.
    (note: not all functions have type hints yet, this is work in progress.)
    - Use the [`str.format`](https://docs.python.org/3/library/stdtypes.html#str.format) over [`printf`](https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting) style formatting.
     E.g. use `"{} {}".format('hello', 'world')` not `"%s %s" % ('hello', 'world')`.
     (note: old code may still use `printf`-formatting, this is work in progress.)

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is
   created.

-  An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be submitted as a `draft`. These may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   Drafts often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.
   
- Add [unit tests](https://github.com/openml/openml-python/tree/develop/tests) and [examples](https://github.com/openml/openml-python/tree/develop/examples) for any new functionality being introduced. 
    - If an unit test contains an upload to the test server, please ensure that it is followed by a file collection for deletion, to prevent the test server from bulking up. For example, `TestBase._mark_entity_for_removal('data', dataset.dataset_id)`, `TestBase._mark_entity_for_removal('flow', (flow.flow_id, flow.name))`.
    - Please ensure that the example is run on the test server by beginning with the call to `openml.config.start_using_configuration_for_example()`.
    - Add the `@pytest.mark.sklearn` marker to your unit tests if they have a dependency on scikit-learn.

-  All tests pass when running `pytest`. On
   Unix-like systems, check with (from the toplevel source folder):

      ```bash
      $ pytest
      ```
   
   For Windows systems, execute the command from an Anaconda Prompt or add `pytest` to PATH before executing the command.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in develop and pass for the PR code.

 - Add your changes to the changelog in the file doc/progress.rst.

 - If any source file is being added to the repository, please add the BSD 3-Clause license to it.


*Note*: We recommend to follow the instructions below to install all requirements locally.
However it is also possible to use the [openml-python docker image](https://github.com/openml/openml-python/blob/main/docker/readme.md) for testing and building documentation.
This can be useful for one-off contributions or when you are experiencing installation issues.

First install openml with its test dependencies by running
  ```bash
  $ pip install -e .[test]
  ```
from the repository folder.
Then configure pre-commit through
 ```bash
 $ pre-commit install
 ```
This will install dependencies to run unit tests, as well as [pre-commit](https://pre-commit.com/).
To run the unit tests, and check their code coverage, run:
  ```bash
  $ pytest --cov=. path/to/tests_for_package
  ```
Make sure your code has good unittest **coverage** (at least 80%).

Pre-commit is used for various style checking and code formatting.
Before each commit, it will automatically run:
 - [black](https://black.readthedocs.io/en/stable/) a code formatter.
   This will automatically format your code.
   Make sure to take a second look after any formatting takes place,
   if the resulting code is very bloated, consider a (small) refactor.
   *note*: If Black reformats your code, the commit will automatically be aborted.
   Make sure to add the formatted files (back) to your commit after checking them.
 - [mypy](https://mypy.readthedocs.io/en/stable/) a static type checker.
   In particular, make sure each function you work on has type hints.
 - [flake8](https://flake8.pycqa.org/en/latest/index.html) style guide enforcement.
   Almost all of the black-formatted code should automatically pass this check,
   but make sure to make adjustments if it does fail.
    
If you want to run the pre-commit tests without doing a commit, run:
```bash
$ make check
```
or on a system without make, like Windows:
```bash
$ pre-commit run --all-files
```
Make sure to do this at least once before your first commit to check your setup works.

Executing a specific unit test can be done by specifying the module, test case, and test.
To obtain a hierarchical list of all tests, run

```bash
$  pytest --collect-only

 <Module 'tests/test_datasets/test_dataset.py'>
   <UnitTestCase 'OpenMLDatasetTest'>
     <TestCaseFunction 'test_dataset_format_constructor'>
     <TestCaseFunction 'test_get_data'>
     <TestCaseFunction 'test_get_data_rowid_and_ignore_and_target'>
     <TestCaseFunction 'test_get_data_with_ignore_attributes'>
     <TestCaseFunction 'test_get_data_with_rowid'>
     <TestCaseFunction 'test_get_data_with_target'>
   <UnitTestCase 'OpenMLDatasetTestOnTestServer'>
     <TestCaseFunction 'test_tagging'>
```

You may then run a specific module, test case, or unit test respectively:
```bash
  $ pytest tests/test_datasets/test_dataset.py
  $ pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest
  $ pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest::test_get_data
```

*NOTE*: In the case the examples build fails during the Continuous Integration test online, please 
fix the first failing example. If the first failing example switched the server from live to test 
or vice-versa, and the subsequent examples expect the other server, the ensuing examples will fail 
to be built as well.

Happy testing!

Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents, tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the doc/ directory.
The resulting HTML files will be placed in ``build/html/`` and are viewable in
a web browser. See the ``README`` file in the ``doc/`` directory for more
information.

For building the documentation, you will need to install a few additional dependencies:
```bash
$ pip install -e .[docs]
```
When dependencies are installed, run
```bash
$ sphinx-build -b html doc YOUR_PREFERRED_OUTPUT_DIRECTORY
```
