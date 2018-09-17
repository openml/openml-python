How to contribute
-----------------

The preferred workflow for contributing to the OpenML python connector is to
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

3. Swith to the ``develop`` branch:

   ```bash
   $ git checkout develop
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b feature/my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` or ``develop`` branch! To make the nature of your pull request easily visible, please perpend the name of the branch with the type of changes you want to merge, such as ``feature`` if it contains a new feature, ``fix`` for a bugfix, ``doc`` for documentation and ``maint`` for other maintenance on the package.

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
   [pep8 style guilde](https://www.python.org/dev/peps/pep-0008/).

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is
   created.

-  An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  All tests pass when running `nosetests`. On
   Unix-like systems, check with (from the toplevel source folder):

      ```bash
      $ nosetests
      ```
   
   For Windows systems, execute the command from an Anaconda Prompt or add `nosetests` to PATH before executing the command.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in develop and pass for the PR code.

 - Add your changes to the changelog in the file doc/progress.rst.


You can also check for common programming errors with the following
tools:

-  Code with good unittest **coverage** (at least 80%), check with:

  ```bash
  $ pip install nose coverage
  $ nosetests --with-coverage path/to/tests_for_package
  ```

-  No pyflakes warnings, check with:

  ```bash
  $ pip install pyflakes
  $ pyflakes path/to/module.py
  ```

-  No PEP8 warnings, check with:

  ```bash
  $ pip install pep8
  $ pep8 path/to/module.py
  ```

Filing bugs
-----------
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

New contributor tips
--------------------

A great way to start contributing to scikit-learn is to pick an item
from the list of [Easy issues](https://github.com/openml/openml-python/issues?q=label%3Aeasy)
in the issue tracker. Resolving these issues allow you to start
contributing to the project without much prior knowledge. Your
assistance in this area will be greatly appreciated by the more
experienced developers as it helps free up their time to concentrate on
other issues.

Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the doc/ directory.
The resulting HTML files will be placed in ``build/html/`` and are viewable in
a web browser. See the ``README`` file in the ``doc/`` directory for more
information.

For building the documentation, you will need
[sphinx](http://sphinx.pocoo.org/),
[matplotlib](http://matplotlib.org/), and
[pillow](http://pillow.readthedocs.io/en/latest/).
[sphinx-bootstrap-theme](https://ryan-roemer.github.io/sphinx-bootstrap-theme/)
