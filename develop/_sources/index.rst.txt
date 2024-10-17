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
    # Download the OpenML task for the pendigits dataset with 10-fold
    # cross-validation.
    task = openml.tasks.get_task(32)
    # Run the scikit-learn model on the task.
    run = openml.runs.run_model_on_task(clf, task)
    # Publish the experiment on OpenML (optional, requires an API key.
    # You can get your own API key by signing up to OpenML.org)
    run.publish()
    print(f'View the run online: {run.openml_url}')

You can find more examples in our :ref:`examples-index`.

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
* :ref:`examples-index`
* :ref:`extensions`
* :ref:`contributing`
* :ref:`progress`

-------------------
Further information
-------------------

* `OpenML documentation <https://docs.openml.org/>`_
* `OpenML client APIs <https://docs.openml.org/APIs/>`_
* `OpenML developer guide <https://docs.openml.org/Contributing/>`_
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

--------------------
Citing OpenML-Python
--------------------

If you use OpenML-Python in a scientific publication, we would appreciate a
reference to the following paper:

| Matthias Feurer, Jan N. van Rijn, Arlind Kadra, Pieter Gijsbers, Neeratyoy Mallik, Sahithya Ravi, Andreas Müller, Joaquin Vanschoren, Frank Hutter
| **OpenML-Python: an extensible Python API for OpenML**
| Journal of Machine Learning Research, 22(100):1−5, 2021
| `https://www.jmlr.org/papers/v22/19-920.html <https://www.jmlr.org/papers/v22/19-920.html>`_

 Bibtex entry::

    @article{JMLR:v22:19-920,
        author  = {Matthias Feurer and Jan N. van Rijn and Arlind Kadra and Pieter Gijsbers and Neeratyoy Mallik and Sahithya Ravi and Andreas MÃ¼ller and Joaquin Vanschoren and Frank Hutter},
        title   = {OpenML-Python: an extensible Python API for OpenML},
        journal = {Journal of Machine Learning Research},
        year    = {2021},
        volume  = {22},
        number  = {100},
        pages   = {1--5},
        url     = {http://jmlr.org/papers/v22/19-920.html}
    }

