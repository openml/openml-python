:orphan:

.. _api:

APIs
****

Top-level Classes
-----------------
.. currentmodule:: openml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   APIConnector
   OpenMLDataset
   OpenMLRun
   OpenMLTask
   OpenMLSplit


Dataset Functions
-----------------
.. currentmodule:: openml.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

    datasets_active
    download_dataset_arff
    download_dataset_description
    download_dataset_features
    download_dataset_qualities
    download_dataset
    download_datasets
    get_cached_datasets
    get_list_of_cached_datasets
    get_dataset_list
    get_cached_dataset
    get_dataset_list

Run Functions
--------------
.. currentmodule:: openml.runs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   construct_description_dictionary
   create_setup_string
   get_version_information
   openml_run
   download_run
   get_cached_run

Task Functions
--------------
.. currentmodule:: openml.tasks

.. autosummary::
   :toctree: generated/
   :template: function.rst

    download_task
    get_task_list
