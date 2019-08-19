.. OpenML documentation master file, created by
   sphinx-quickstart on Wed Nov 26 10:46:10 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
OpenML
======

**Collaborative Machine Learning in Python**

Welcome to the documentation of the OpenML Python API, a connector to the
collaborative machine learning platform `OpenML.org <https://www.openml.org>`_.
The OpenML Python package allows to use datasets and tasks from OpenML together
with scikit-learn and share the results online.

-------
Example
-------

.. code:: python

    import openml
    from sklearn import impute, tree, pipeline

    # Define a scikit-learn classifier or pipeline
    clf = pipeline.Pipeline(
        steps=[
            ('imputer', impute.SimpleImputer()),
            ('estimator', tree.DecisionTreeClassifier())
        ]
    )
    # Download the OpenML task for the german credit card dataset with 10-fold
    # cross-validation.
    task = openml.tasks.get_task(31)
    # Run the scikit-learn model on the task.
    run = openml.runs.run_model_on_task(clf, task)
    # Publish the experiment on OpenML (optional, requires an API key.
    # You can get your own API key by signing up to OpenML.org)
    run.publish()
    print('View the run online: %s/run/%d' % (openml.config.server, run.run_id))

You can find more examples in our `examples gallery <examples/index.html>`_.

----------------------------
How to get OpenML for python
----------------------------
You can install the OpenML package via `pip`:

.. code:: bash

    pip install openml

For more advanced installation information, please see the
:ref:`installation` section.

-------
Content
-------

* :ref:`usage`
* :ref:`api`
* `Examples <examples/index.html>`_
* :ref:`contributing`
* :ref:`progress`

-------------------
Further information
-------------------

* `OpenML documentation <https://docs.openml.org/>`_
* `OpenML client APIs <https://docs.openml.org/APIs/>`_
* `OpenML developer guide <https://docs.openml.org/developers/>`_
* `Contact information <https://www.openml.org/contact>`_
* `Citation request <https://www.openml.org/cite>`_
* `OpenML blog <https://medium.com/open-machine-learning>`_
* `OpenML twitter account <https://twitter.com/open_ml>`_

------------
Contributing
------------

Contribution to the OpenML package is highly appreciated. The OpenML package
currently has a 1/4 position for the development and all help possible is
needed to extend and maintain the package, create new examples and improve
the usability. Please see the :ref:`contributing` page for more information.
