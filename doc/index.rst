.. OpenML documentation master file, created by
   sphinx-quickstart on Wed Nov 26 10:46:10 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
OpenML
======

Welcome to the documentation of the OpenML Python API, a connector to the
collaborative machine learning platform `OpenML.org <https://www.openml.org>`_.
The OpenML Python package allows to use datasets and tasks from OpenML together
with scikit-learn and share the results online.

-------
Example
-------

.. code:: python

    # Define a scikit-learn pipeline
    clf = sklearn.pipeline.Pipeline(
        steps=[
            ('imputer', sklearn.preprocessing.Imputer()),
            ('estimator', sklearn.tree.DecisionTreeClassifier())
        ]
    )
    # Download the OpenML task for the german credit card dataset with 10-fold
    # cross-validation.
    task = openml.tasks.get_task(31)
    # Set the OpenML API Key which is required to upload the runs.
    # You can get your own API by signing up to OpenML.org.
    openml.config.apikey = 'ABC'
    # Run the scikit-learn model on the task (requires an API key).
    run = openml.runs.run_model_on_task(task, clf)
    # Publish the experiment on OpenML (optional, requires an API key).
    run.publish()
    print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))


------------
Introduction
------------

How to get OpenML for python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can install the OpenML package via `pip`:

.. code:: bash

    pip install openml
    

Installation via GitHub (for developers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The package source code is available from
`github <https://github.com/openml/openml-python>`_.

.. code:: bash

    git clone https://github.com/openml/openml-python.git


Once you cloned the package, change into the new directory ``python`` and
execute

.. code:: bash

    python setup.py install

Testing
~~~~~~~

From within the directory of the cloned package, execute

.. code:: bash

    python setup.py test

Usage
~~~~~

* :ref:`usage`
* :ref:`api`
* :ref:`developing`

Contributing
~~~~~~~~~~~~

Contribution to the OpenML package is highly appreciated. Currently,
there is a lot of work left on implementing API calls,
testing them and providing examples to allow new users to easily use the
OpenML package. See the :ref:`progress` page for open tasks.

Please contact `Matthias <http://aad.informatik.uni-freiburg.de/people/feurer/index.html>`_
prior to start working on an issue or missing feature to avoid duplicate work
. Please check the current implementations of the API calls and the method
