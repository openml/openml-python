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

-------------
First Example
-------------

.. code:: python

    import openml
    from sklearn import preprocessing, tree, pipeline

    # Set the OpenML API Key which is required to upload your runs.
    # You can get your own API by signing up to OpenML.org.
    openml.config.apikey = 'ABC'

    # Define a scikit-learn classifier or pipeline
    clf = pipeline.Pipeline(
        steps=[
            ('imputer', preprocessing.Imputer()),
            ('estimator', tree.DecisionTreeClassifier())
        ]
    )

    # Download the OpenML task for the german credit card dataset with 10-fold
    # cross-validation.
    task = openml.tasks.get_task(31)

    # Run the scikit-learn model on the task.
    run = openml.runs.run_model_on_task(clf, task)

    # Publish the experiment on OpenML (optional, requires an API key).
    run.publish()
    print('View the run online: %s/run/%d' % (openml.config.server, run.run_id))

----------------------------
How to get OpenML for Python
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
Further Information
-------------------

For more information on what you can do with OpenML and how to engage in collaborative machine learning, check out the tutorials below. To reach out to us for more informaiton regarding events, citations and updates from the OpenML team, visit our Twitter or Blog.

Get Started with OpenML
~~~~~~~~~~~~~~~~~~~~~~~

* `OpenML Documentation <https://docs.openml.org/>`_
* `OpenML Client APIs <https://docs.openml.org/APIs/>`_
* `OpenML Developer Guide <https://docs.openml.org/developers/>`_
* `OpenML Tutorials <https://openml.github.io/openml-tutorial/>`_

Contact Us
~~~~~~~~~~~~~~~~~
* `Contact Information <https://www.openml.org/contact>`_
* `Citation Request <https://www.openml.org/cite>`_
* `OpenML Blog <htt ps://medium.com/open-machine-learning>`_
* `OpenML Twitter Account <https://twitter.com/open_ml>`_

------------
Contributing
------------

Contribution to the OpenML package is highly appreciated. The OpenML package
currently has a 1/4 position for the development and all help possible is
needed to extend and maintain the package, create new examples and improve
the usability. Please see the :ref:`contributing` page for more information.
