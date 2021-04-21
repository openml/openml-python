:orphan:

.. _contributing:

============
Contributing
============

Contribution to the OpenML package is highly appreciated in all forms.
In particular, a few ways to contribute to openml-python are:

 * A direct contribution to the package, by means of improving the
   code, documentation or examples. To get started, see `this file <https://github.com/openml/openml-python/blob/master/CONTRIBUTING.md>`_
   with details on how to set up your environment to develop for openml-python.

 * A contribution to an openml-python extension. An extension package allows OpenML to interface
   with a machine learning package (such as scikit-learn or keras). These extensions
   are hosted in separate repositories and may have their own guidelines.
   For more information, see the :ref:`extensions` below.

 * Bug reports. If something doesn't work for you or is cumbersome, please open a new issue to let
   us know about the problem. See `this section <https://github.com/openml/openml-python/blob/master/CONTRIBUTING.md#user-content-reporting-bugs>`_.

 * `Cite OpenML <https://www.openml.org/cite>`_ if you use it in a scientific publication.

 * Visit one of our `hackathons <https://meet.openml.org/>`_.

 * Contribute to another OpenML project, such as `the main OpenML project <https://github.com/openml/OpenML/blob/master/CONTRIBUTING.md>`_.

.. _extensions:

Connecting new machine learning libraries
=========================================

Content of the Library
~~~~~~~~~~~~~~~~~~~~~~

To leverage support from the community and to tap in the potential of OpenML, interfacing
with popular machine learning libraries is essential. However, the OpenML-Python team does
not have the capacity to develop and maintain such interfaces on its own. For this, we
have built an extension interface to allows others to contribute back. Building a suitable
extension for therefore requires an understanding of the current OpenML-Python support.

The :ref:`sphx_glr_examples_20_basic_simple_flows_and_runs_tutorial.py` tutorial
shows how scikit-learn currently works with OpenML-Python as an extension. The *sklearn*
extension packaged with the `openml-python <https://github.com/openml/openml-python>`_
repository can be used as a template/benchmark to build the new extension.


API
+++
* The extension scripts must import the `openml` package and be able to interface with
  any function from the OpenML-Python :ref:`api`.
* The extension has to be defined as a Python class and must inherit from
  :class:`openml.extensions.Extension`.
* This class needs to have all the functions from `class Extension` overloaded as required.
* The redefined functions should have adequate and appropriate docstrings. The
  `Sklearn Extension API :class:`openml.extensions.sklearn.SklearnExtension.html`
  is a good benchmark to follow.


Interfacing with OpenML-Python
++++++++++++++++++++++++++++++
Once the new extension class has been defined, the openml-python module to
:meth:`openml.extensions.register_extension` must be called to allow OpenML-Python to
interface the new extension.


Hosting the library
~~~~~~~~~~~~~~~~~~~

Each extension created should be a stand-alone repository, compatible with the
`OpenML-Python repository <https://github.com/openml/openml-python>`_.
The extension repository should work off-the-shelf with *OpenML-Python* installed.

Create a `public Github repo <https://docs.github.com/en/github/getting-started-with-github/create-a-repo>`_
with the following directory structure:

::

| [repo name]
|    |-- [extension name]
|    |    |-- __init__.py
|    |    |-- extension.py
|    |    |-- config.py (optionally)



Recommended
~~~~~~~~~~~
* Test cases to keep the extension up to date with the `openml-python` upstream changes.
* Documentation of the extension API, especially if any new functionality added to OpenML-Python's
  extension design.
* Examples to show how the new extension interfaces and works with OpenML-Python.
* Create a PR to add the new extension to the OpenML-Python API documentation.


Happy contributing!
