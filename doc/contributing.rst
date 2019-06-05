:orphan:

.. _contributing:


============
Contributing
============

Contribution to the OpenML package is highly appreciated. Currently,
there is a lot of work left on implementing API calls,
testing them and providing examples to allow new users to easily use the
OpenML package. See the :ref:`issues` section for open tasks.

Please mark yourself as contributor in a github issue if you start working on
something to avoid duplicate work. If you're part of the OpenML organization
you can use github's assign feature, otherwise you can just leave a comment.

.. _scope:

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

.. _issues:

Open issues and potential todos
===============================

We collect open issues and feature requests in an `issue tracker on github <https://github.com/openml/openml-python/issues>`_.
The issue tracker contains issues marked as *Good first issue*, which shows
issues which are good for beginners. We also maintain a somewhat up-to-date
`roadmap <https://github.com/openml/openml-python/issues/410>`_ which
contains longer-term goals.

.. _how_to_contribute:

How to contribute
=================

There are many ways to contribute to the development of the OpenML python
connector and OpenML in general. We welcome all kinds of contributions,
especially:

* Source code which fixes an issue, improves usability or implements a new
  feature.
* Improvements to the documentation, which can be found in the ``doc``
  directory.
* New examples - current examples can be found in the ``examples`` directory.
* Bug reports - if something doesn't work for you or is cumbersome, please
  open a new issue to let us know about the problem.
* Use the package and spread the word.
* `Cite OpenML <https://www.openml.org/cite>`_ if you use it in a scientific
  publication.
* Visit one of our `hackathons <https://meet.openml.org/>`_.
* Check out how to `contribute to the main OpenML project <https://github.com/openml/OpenML/blob/master/CONTRIBUTING.md>`_.

Contributing code
~~~~~~~~~~~~~~~~~

Our guidelines on code contribution can be found in `this file <https://github.com/openml/openml-python/blob/master/CONTRIBUTING.md>`_.

.. _installation:

Installation
============

Installation from github
~~~~~~~~~~~~~~~~~~~~~~~~

The package source code is available from
`github <https://github.com/openml/openml-python>`_ and can be obtained with:

.. code:: bash

    git clone https://github.com/openml/openml-python.git


Once you cloned the package, change into the new directory.
If you are a regular user, install with

.. code:: bash

    pip install -e .

If you are a contributor, you will also need to install test dependencies

.. code:: bash

    pip install -e ".[test]"


Testing
=======

From within the directory of the cloned package, execute:

.. code:: bash

    pytest tests/

Executing a specific test can be done by specifying the module, test case, and test.
To obtain a hierarchical list of all tests, run

.. code:: bash

    pytest --collect-only

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

    pytest tests/test_datasets/test_dataset.py

To run a specific unit test case, add the test case name, for instance:

.. code:: bash

    pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest

To run a specific unit test, add the test name, for instance:

.. code:: bash

    pytest tests/test_datasets/test_dataset.py::OpenMLDatasetTest::test_get_data

Happy testing!


Connecting new machine learning libraries
=========================================

Coming soon - please stay tuned!

