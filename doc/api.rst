:orphan:

.. _api:

API
***

Modules
=======

:mod:`openml.datasets`
----------------------
.. automodule:: openml.datasets
    :no-members:
    :no-inherited-members:

Dataset Classes
~~~~~~~~~~~~~~~

.. currentmodule:: openml.datasets

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OpenMLDataFeature
   OpenMLDataset

Dataset Functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: openml.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

    attributes_arff_from_df
    check_datasets_active
    create_dataset
    delete_dataset
    get_dataset
    get_datasets
    list_datasets
    list_qualities
    status_update
    edit_dataset
    fork_dataset

:mod:`openml.evaluations`
-------------------------
.. automodule:: openml.evaluations
    :no-members:
    :no-inherited-members:

Evaluations Classes
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: openml.evaluations

.. autosummary::
   :toctree: generated/
   :template: class.rst

    OpenMLEvaluation

Evaluations Functions
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: openml.evaluations

.. autosummary::
   :toctree: generated/
   :template: function.rst

   list_evaluations
   list_evaluation_measures
   list_evaluations_setups

:mod:`openml.flows`: Flow Functions
-----------------------------------
.. automodule:: openml.flows
    :no-members:
    :no-inherited-members:

Flow Classes
~~~~~~~~~~~~

.. currentmodule:: openml.flows

.. autosummary::
   :toctree: generated/
   :template: class.rst

    OpenMLFlow

Flow Functions
~~~~~~~~~~~~~~

.. currentmodule:: openml.flows

.. autosummary::
   :toctree: generated/
   :template: function.rst

    assert_flows_equal
    delete_flow
    flow_exists
    get_flow
    list_flows

:mod:`openml.runs`: Run Functions
----------------------------------
.. automodule:: openml.runs
    :no-members:
    :no-inherited-members:

Run Classes
~~~~~~~~~~~

.. currentmodule:: openml.runs

.. autosummary::
   :toctree: generated/
   :template: class.rst

    OpenMLRun

Run Functions
~~~~~~~~~~~~~

.. currentmodule:: openml.runs

.. autosummary::
   :toctree: generated/
   :template: function.rst

    delete_run
    get_run
    get_runs
    get_run_trace
    initialize_model_from_run
    initialize_model_from_trace
    list_runs
    run_model_on_task
    run_flow_on_task
    run_exists

:mod:`openml.setups`: Setup Functions
-------------------------------------
.. automodule:: openml.setups
    :no-members:
    :no-inherited-members:

Setup Classes
~~~~~~~~~~~~~

.. currentmodule:: openml.setups

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OpenMLParameter
   OpenMLSetup

Setup Functions
~~~~~~~~~~~~~~~

.. currentmodule:: openml.setups

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_setup
    initialize_model
    list_setups
    setup_exists

:mod:`openml.study`: Study Functions
------------------------------------
.. automodule:: openml.study
    :no-members:
    :no-inherited-members:

Study Classes
~~~~~~~~~~~~~

.. currentmodule:: openml.study

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OpenMLBenchmarkSuite
   OpenMLStudy

Study Functions
~~~~~~~~~~~~~~~

.. currentmodule:: openml.study

.. autosummary::
   :toctree: generated/
   :template: function.rst

   attach_to_study
   attach_to_suite
   create_benchmark_suite
   create_study
   delete_study
   delete_suite
   detach_from_study
   detach_from_suite
   get_study
   get_suite
   list_studies
   list_suites
   update_study_status
   update_suite_status

:mod:`openml.tasks`: Task Functions
-----------------------------------
.. automodule:: openml.tasks
    :no-members:
    :no-inherited-members:

Task Classes
~~~~~~~~~~~~

.. currentmodule:: openml.tasks

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OpenMLClassificationTask
   OpenMLClusteringTask
   OpenMLLearningCurveTask
   OpenMLRegressionTask
   OpenMLSplit
   OpenMLSupervisedTask
   OpenMLTask
   TaskType

Task Functions
~~~~~~~~~~~~~~

.. currentmodule:: openml.tasks

.. autosummary::
   :toctree: generated/
   :template: function.rst

    create_task
    delete_task
    get_task
    get_tasks
    list_tasks

.. _api_extensions:

Extensions
==========

.. automodule:: openml.extensions
    :no-members:
    :no-inherited-members:

Extension Classes
-----------------

.. currentmodule:: openml.extensions

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Extension
   sklearn.SklearnExtension

Extension Functions
-------------------

.. currentmodule:: openml.extensions

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_extension_by_flow
    get_extension_by_model
    register_extension

