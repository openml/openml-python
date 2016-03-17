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

    >>> apikey = 'Your API key'
    >>> connector = APIConnector(apikey=apikey)

The :class:`~openml.apiconnector.APIConnector` will create a cache directory
and manage all your queries to the OpenML server.

You can also configure the OpenML package, e.g. change the cache directory.
Information about the configuration is in the
`OpenML client API description <https://github
.com/openml/OpenML/wiki/Client-API>`_.

Working with datasets
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> dataset_id = 31
    >>> dataset = connector.get_dataset(dataset_id)

Attributes of the dataset are stored as member variables:

.. code:: python

    >>> dataset.name
    u'credit-g'
    >>> dataset.default_target_attribute
    u'class'

Data can be loaded in the following ways:

.. code:: python

    >>> X = dataset.get_dataset()

returns the dataset as a np.ndarray. In case the data is sparse,
a scipy.sparse.csr matrix is returned.

Most times, having only the X matrix is not enough. Two very useful arguments
are `target` and `return_categorical_indicator`. `target` makes `get_dataset
()` return `X` and `y` seperate; `return_categorical_indicator` makes
`get_dataset()` return a boolean array which indicate which attributes are
categorical (and should be one hot encoded.)

.. code:: python

    >>> X, y, categorical = dataset.get_dataset(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True)

In case you are working with `scikit-learn
<http://scikit-learn>`_, you can use this data right away:

.. code:: python

    >>> from sklearn import preprocessing, ensemble
    >>> enc = preprocessing.OneHotEncoder(categorical_features=categorical)
    OneHotEncoder(categorical_features=[True, False, True, True, False, True,
        True, False, True, True, False, True, False, True, True, False, True,
        False, True, True], dtype=<type 'float'>, n_values='auto',
        sparse=True)
    >>> X = enc.fit_transform(X).todense()
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