.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_datasets_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_datasets_tutorial.py:


========
Datasets
========

How to list and download datasets.


.. code-block:: default


    import openml
    import pandas as pd







List datasets
=============


.. code-block:: default


    openml_list = openml.datasets.list_datasets()  # returns a dict

    # Show a nice table with some key data properties
    datalist = pd.DataFrame.from_dict(openml_list, orient='index')
    datalist = datalist[[
        'did', 'name', 'NumberOfInstances',
        'NumberOfFeatures', 'NumberOfClasses'
    ]]

    print("First 10 of %s datasets..." % len(datalist))
    datalist.head(n=10)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    First 10 of 15672 datasets...


Exercise 1
**********

* Find datasets with more than 10000 examples.
* Find a dataset called 'eeg_eye_state'.
* Find all datasets with more than 50 classes.


.. code-block:: default

    datalist[datalist.NumberOfInstances > 10000
             ].sort_values(['NumberOfInstances']).head(n=20)







.. code-block:: default

    datalist.query('name == "eeg-eye-state"')







.. code-block:: default

    datalist.query('NumberOfClasses > 50')







Download datasets
=================


.. code-block:: default


    # This is done based on the dataset ID ('did').
    dataset = openml.datasets.get_dataset(68)
    # NOTE: Dataset 68 exists on the test server https://test.openml.org/d/68

    # Print a summary
    print("This is dataset '%s', the target feature is '%s'" %
          (dataset.name, dataset.default_target_attribute))
    print("URL: %s" % dataset.url)
    print(dataset.description[:500])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    This is dataset 'eeg-eye-state', the target feature is 'Class'
    URL: https://test.openml.org/data/v1/download/68/eeg-eye-state.arff
    **Author**: Oliver Roesler, it12148'@'lehre.dhbw-stuttgart.de  
    **Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State), Baden-Wuerttemberg, Cooperative State University (DHBW), Stuttgart, Germany  
    **Please cite**:   

    All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video fr


Get the actual data.

The dataset can be returned in 2 possible formats: as a NumPy array, a SciPy
sparse matrix, or as a Pandas DataFrame (or SparseDataFrame). The format is
controlled with the parameter ``dataset_format`` which can be either 'array'
(default) or 'dataframe'. Let's first build our dataset from a NumPy array
and manually create a dataframe.


.. code-block:: default

    X, y, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute,
        return_attribute_names=True,
    )
    eeg = pd.DataFrame(X, columns=attribute_names)
    eeg['class'] = y
    print(eeg[:10])




.. code-block:: pytb

    Traceback (most recent call last):
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 394, in _memory_usage
        out = func()
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 382, in __call__
        exec(self.code, self.globals)
      File "/Users/michaelmmeskhi/Documents/GitHub/openml-python/examples/datasets_tutorial.py", line 67, in <module>
        return_attribute_names=True,
    TypeError: get_data() got an unexpected keyword argument 'dataset_format'




Instead of manually creating the dataframe, you can already request a
dataframe with the correct dtypes.


.. code-block:: default

    X, y = dataset.get_data(target=dataset.default_target_attribute,
                            dataset_format='dataframe')
    print(X.head())
    print(X.info())


Sometimes you only need access to a dataset's metadata.
In those cases, you can download the dataset without downloading the
data file. The dataset object can be used as normal.
Whenever you use any functionality that requires the data,
such as `get_data`, the data will be downloaded.


.. code-block:: default

    dataset = openml.datasets.get_dataset(68, download_data=False)
    # NOTE: Dataset 68 exists on the test server https://test.openml.org/d/68


Exercise 2
**********
* Explore the data visually.


.. code-block:: default

    eegs = eeg.sample(n=1000)
    _ = pd.plotting.scatter_matrix(
        eegs.iloc[:100, :4],
        c=eegs[:100]['class'],
        figsize=(10, 10),
        marker='o',
        hist_kwds={'bins': 20},
        alpha=.8,
        cmap='plasma'
    )


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.782 seconds)


.. _sphx_glr_download_examples_datasets_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: datasets_tutorial.py <datasets_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: datasets_tutorial.ipynb <datasets_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
