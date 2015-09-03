:orphan:

.. _usage:

Basic Usage
***********

Connecting to the OpenML server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenML server can only be accessed by users who have signed up to the OpenML
platform. If you don't have an account yet,
`sign up now <http://openml.org/register>`_.

.. code:: python

    >>> from openml.apiconnector import APIConnector

    >>> username = "Your OpenML username"
    >>> password = "Your OpenML password"
    >>> connector = APIConnector(username=username, password=password)

The :class:`~openml.apiconnector.APIConnector` will create a cache directory
and authenticate you at the OpenML server. By this you obtain a session key,
which is valid for one hour.

You can also configure the OpenML package, e.g. change the cache directory.
Information about the configuration is in the
`OpenML client API description <https://github
.com/openml/OpenML/wiki/Client-API>`_.

Working with datasets
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> dataset_id = 31
    >>> dataset = connector.download_dataset(1)

Attributes of the dataset are stored as member variables:

.. code:: python

    >>> dataset.name
    u'credit-g'
    >>> dataset.default_target_attribute
    u'class'

Data can be loaded in the following ways:

.. code:: python

    >>> pd, categorical = dataset.get_pandas()

returns the dataset as a pandas.DataFrame and a list of booleans,
indicating which attributes are categorical. Categorical attributes are
already encoded as integers.

.. code:: python

    >>> X, y, categorical = dataset.get_pandas()

returns the dataset split into X and y, as well as a list indicating which
attributes are categorical. In case you are working with `scikit-learn
<http://scikit-learn>`_, you can use this data right away:

.. code:: python

    >>> from sklearn import preprocessing, ensemble
    >>> enc = preprocessing.OneHotEncoder(categorical_features=categorical)
    OneHotEncoder(categorical_features=[True, False, True, True, False, True,
        True, False, True, True, False, True, False, True, True, False, True,
        False, True, True], dtype=<type 'float'>, n_values='auto',
        sparse=True)
    >>> X = enc.transform(X).todense()
    >>> clf = ensemble.RandomForestClassifier()
    >>> clf.fit(X, y)
    RandomForestClassifier(bootstrap=True, compute_importances=None,
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
            min_samples_split=2, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0)

Working with tasks
~~~~~~~~~~~~~~~~~~

Using the cache
~~~~~~~~~~~~~~~

Large scale experiments
~~~~~~~~~~~~~~~~~~~~~~~