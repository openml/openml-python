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
   OpenMLFlow


Dataset Functions
-----------------
.. currentmodule:: openml.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

    check_datasets_active
    get_dataset_description
    get_dataset_features
    get_dataset_qualities
    get_dataset
    get_datasets
    list_datasets

Run Functions
-------------
.. currentmodule:: openml.runs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   run_task
   get_run
   list_runs
   list_runs_by_flow
   list_runs_by_tag
   list_runs_by_task
   list_runs_by_uploader
   list_runs_by_filters

Task Functions
--------------
.. currentmodule:: openml.tasks

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_task
    list_tasks

Flow Functions
--------------
.. currentmodule:: openml.flow

.. autosummary::
   :toctree: generated/
   :template: function.rst
 
