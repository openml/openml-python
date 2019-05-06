:orphan:

.. _contributing:


============
Contributing
============

Contribution to the OpenML package is highly appreciated. Currently,
there is a lot of work left on implementing API calls, testing them and providing examples to allow new users to easily use the OpenML package. See the :ref:`issues` section for open tasks.

Please mark yourself as contributor in a github issue if you start working on
something to avoid duplicate work. If you're part of the OpenML organization
you can use github's assign feature, otherwise you can just leave a comment.

.. _tips:


New contributor tips
~~~~~~~~~~~~~~~~~~~~

A great way to start contributing to openml-python is to pick an item
from the list of `Good First Issues <https://github.com/openml/openml-python/labels/Good%20first%20issue>`_ in the issue tracker. Resolving these issues allow you to start
contributing to the project without much prior knowledge. Your assistance in this area will be greatly appreciated by the more experienced developers as it helps free up their time to concentrate on.

.. _howto:


How to Contribute
~~~~~~~~~~~~~~~~~

There are many ways to contribute to the development of the OpenML python
connector and OpenML in general. We welcome all kinds of contributions,
especially:

* Source code which fixes an issue, improves usability or implements a new feature.
* Improvements to the documentation, which can be found in the ``doc`` directory.
* New examples - current examples can be found in the ``examples`` directory.
* Bug reports - if something doesn't work for you or is cumbersome, please open a new issue to let us know about the problem.
* Use the package and spread the word.
* `Cite OpenML <https://www.openml.org/cite>`_ if you use it in a scientific publication.
* Visit one of our `hackathons <https://meet.openml.org/>`_.
* Check out how to `contribute to the main OpenML project <https://github.com/openml/OpenML/blob/master/CONTRIBUTING.md>`_.

The preferred workflow for contributing to the OpenML python connector is to fork the `main repository <https://github.com/openml/openml-python>`_ on GitHub, clone, check out the branch ``develop``, and develop on a new branch branch. Steps:

1. Fork the `project repository <https://github.com/openml/openml-python>`_ by clicking on the 'Fork' button near the top right of the page. This creates a copy of the code under your GitHub user account. For more details on how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork of the openml-python repo from your GitHub account to your local disk:

.. code:: bash

    $ git clone git@github.com:YourLogin/openml-python.git
    $ cd openml-python

3. Switch to the develop branch:

.. code:: bash
	
    $ git checkout develop

4. Create a feature branch to hold your development changes:

.. code:: bash

    $ git checkout -b feature/my-feature
	
Always use a feature branch. It's good practice to never work on the master or develop branch! To make the nature of your pull request easily visible, please prepend the name of the branch with the type of changes you want to merge, such as feature if it contains a new feature, fix for a bugfix, doc for documentation and maint for other maintenance on the package.

Develop the feature on your feature branch. Add changed files using git add and then git commit files:

.. code:: bash

    $ git add modified_files
    $ git commit
	
to record your changes in Git, then push the changes to your GitHub account with:

.. code:: bash

    $ git push -u origin my-feature

Follow these instructions to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the Git documentation on the web, or ask a friend or another contributor for help.)

Our guidelines on code contribution can be found in `this file <https://github.com/openml/openml-python/blob/master/CONTRIBUTING.md>`_.

.. _issues:


Open issues and potential todos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We collect open issues and feature requests in an `issue tracker on github <https://github.com/openml/openml-python/issues>`_.
The issue tracker contains issues marked as *Good first issue*, which shows
issues which are good for beginners. We also maintain a somewhat up-to-date
`roadmap <https://github.com/openml/openml-python/issues/410>`_ which
contains longer-term goals.


.. _pull:


Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

We recommended that your contribution complies with the
following rules before you submit a pull request:

*  Follow the `pep8 style guide <https://www.python.org/dev/peps/pep-0008/>`_.

With the following exceptions or additions:

* The max line length is 100 characters instead of 80.
* When creating a multi-line expression with binary operators, break before the operator.
* Add type hints to all function signatures. (note: not all functions have type hints yet, this is work in progress.)
* Use the `str.format <https://docs.python.org/3/library/stdtypes.html#str.format>`_ over `printf <https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting>`_ style formatting.

E.g. use ``"{} {}".format('hello', 'world') not "%s %s" % ('hello', 'world')``. (note: old code may still use `printf`-formatting, this is work in progress.)

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

*  An incomplete contribution -- where you expect to do more work before receiving a full review -- should be prefixed `[WIP]` (to indicate a work in progress) and changed to `[MRG]` when it matures. WIPs may be useful to: indicate you are working on something to avoid duplicated work, request broad review of functionality or API, or seek collaborators. WIPs often benefit from the inclusion of a `task list <https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments>`_ in the PR description.

*  All tests pass when running ``pytest``. On Unix-like systems, check with (from the toplevel source folder):

.. code:: bash

    $ pytest
   
For Windows systems, execute the command from an Anaconda Prompt or add ``pytest`` to PATH before executing the command.

*  Documentation and high-coverage tests are necessary for enhancements to be accepted. Bug-fixes or new features should be provided with `non-regression tests <https://en.wikipedia.org/wiki/Non-regression_testing>`_. These tests verify the correct behavior of the fix or feature. In this manner, further modifications on the code base are granted to be consistent with the desired behavior. For the Bug-fixes case, at the time of the PR, this tests should fail for the code base in develop and pass for the PR code.

* Add your changes to the changelog in the file doc/progress.rst.


You can also check for common programming errors with the following tools:

*  Code with good unittest **coverage** (at least 80%), check with:

.. code:: bash

    $ pip install pytest pytest-cov
    $ pytest --cov=. path/to/tests_for_package

*  No style warnings, check with:

.. code:: bash

    $ pip install flake8
    $ flake8 --ignore E402,W503 --show-source --max-line-length 100

*  No mypy (typing) issues, check with:

.. code:: bash

    $ pip install mypy
    $ mypy openml --ignore-missing-imports --follow-imports skip


.. _scope:


====================
Scope of the package
====================

The scope of the OpenML python package is to provide a python interface to
the OpenML platform which integrates well with pythons scientific stack, most
notably `numpy <http://www.numpy.org/>`_ and `scipy <https://www.scipy.org/>`_.
To reduce opportunity costs and demonstrate the usage of the package, it also
implements an interface to the most popular machine learning package written
in python, `scikit-learn <http://scikit-learn.org/stable/index.html>`_.
Thereby it will automatically be compatible with many machine learning
libraries written in Python.

We aim to keep the package as light-weight as possible and we will try to
keep the number of potential installation dependencies as low as possible.
Therefore, the connection to other machine learning libraries such as
*pytorch*, *keras* or *tensorflow* should not be done directly inside this
package, but in a separate package using the OpenML python connector.


.. _installation:


============
Installation
============

Installation from github
~~~~~~~~~~~~~~~~~~~~~~~~

The package source code is available from
`github <https://github.com/openml/openml-python>`_ and can be obtained with:

.. code:: bash

    $ git clone https://github.com/openml/openml-python.git


Once you cloned the package, change into the new directory.
If you are a regular user, install with

.. code:: bash

    $ pip install -e .

If you are a contributor, you will also need to install test dependencies

.. code:: bash

    $ pip install -e ".[test]"


.. _testing:


Testing
~~~~~~~

From within the directory of the cloned package, execute:

.. code:: bash

    $ pytest tests/

Executing a specific test can be done by specifying the module, test case, and test.
To obtain a hierarchical list of all tests, run

.. code:: bash

    $ pytest --collect-only

.. code:: bash

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


To run a specific module, add the module name, for instance:

.. code:: bash

    $ pytest tests/test_datasets/test_dataset.py

To run a specific unit test case, add the test case name, for instance:

.. code:: bash

    $ pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest

To run a specific unit test, add the test name, for instance:

.. code:: bash

    $ pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest::test_get_data

Happy testing!


.. _newmllib:


Connecting new machine learning libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon - please stay tuned!

