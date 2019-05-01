.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_introduction_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_introduction_tutorial.py:


Introduction
===================

An introduction to OpenML, followed up by a simple example.

OpenML is an online collaboration platform for machine learning which allows
you to:

* Find or share interesting, well-documented datasets
* Define research / modelling goals (tasks)
* Explore large amounts of machine learning algorithms, with APIs in Java, R, Python
* Log and share reproducible experiments, models, results
* Works seamlessly with scikit-learn and other libraries
* Large scale benchmarking, compare to state of the art

Installation
^^^^^^^^^^^^
Installation is done via ``pip``:

.. code:: bash

    pip install openml

For further information, please check out the installation guide at
https://openml.github.io/openml-python/master/contributing.html#installation

Authentication
^^^^^^^^^^^^^^

The OpenML server can only be accessed by users who have signed up on the
OpenML platform. If you don’t have an account yet, sign up now.
You will receive an API key, which will authenticate you to the server
and allow you to download and upload datasets, tasks, runs and flows.

* Create an OpenML account (free) on http://www.openml.org.
* After logging in, open your account page (avatar on the top right)
* Open 'Account Settings', then 'API authentication' to find your API key.

There are two ways to authenticate:

* Create a plain text file **~/.openml/config** with the line
  **'apikey=MYKEY'**, replacing **MYKEY** with your API key. The config
  file must be in the directory ~/.openml/config and exist prior to
  importing the openml module.
* Run the code below, replacing 'YOURKEY' with your API key.


.. code-block:: default

    import openml
    from sklearn import neighbors

    # Uncomment and set your OpenML key. Don't share your key with others.
    # openml.config.apikey = 'YOURKEY'







Caching
^^^^^^^
When downloading datasets, tasks, runs and flows, they will be cached to
retrieve them without calling the server later. As with the API key,
the cache directory can be either specified through the config file or
through the API:

* Add the  line **cachedir = 'MYDIR'** to the config file, replacing
  'MYDIR' with the path to the cache directory. By default, OpenML
  will use **~/.openml/cache** as the cache directory.
* Run the code below, replacing 'YOURDIR' with the path to the cache directory.


.. code-block:: default


    # Uncomment and set your OpenML cache directory
    # import os
    # openml.config.cache_directory = os.path.expanduser('YOURDIR')







Simple Example
^^^^^^^^^^^^^^
Download the OpenML task for the eeg-eye-state.


.. code-block:: default

    task = openml.tasks.get_task(403)
    data = openml.datasets.get_dataset(task.dataset_id)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)
    # Publish the experiment on OpenML (optional, requires an API key).
    # For this tutorial, our configuration publishes to the test server
    # as to not pollute the main server.
    myrun = run.publish()
    print("kNN on %s: http://test.openml.org/r/%d" % (data.name, myrun.run_id))



.. code-block:: pytb

    Traceback (most recent call last):
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 394, in _memory_usage
        out = func()
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 382, in __call__
        exec(self.code, self.globals)
      File "/Users/michaelmmeskhi/Documents/GitHub/openml-python/examples/introduction_tutorial.py", line 80, in <module>
        run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/openml-0.6.0-py3.6.egg/openml/runs/functions.py", line 36, in run_model_on_task
        flow = sklearn_to_flow(model)
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/openml-0.6.0-py3.6.egg/openml/flows/sklearn_converter.py", line 81, in sklearn_to_flow
        raise TypeError(o, type(o))
    TypeError: (<openml.tasks.task.OpenMLTask object at 0x112d00e48>, <class 'openml.tasks.task.OpenMLTask'>)





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.531 seconds)


.. _sphx_glr_download_examples_introduction_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: introduction_tutorial.py <introduction_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: introduction_tutorial.ipynb <introduction_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
