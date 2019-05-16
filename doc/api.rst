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

   OpenMLDataset
   OpenMLRun
   OpenMLTask
   OpenMLSplit
   OpenMLFlow
   OpenMLEvaluation

.. _api_extensions:

Extensions
----------

.. currentmodule:: openml.extensions

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Extension
   sklearn.SklearnExtension

.. currentmodule:: openml.extensions

.. autosummary::
   :toctree: generated/
   :template: function.rst

    register_extension
    get_extension_by_model
    get_extension_by_flow

Modules
-------

:mod:`openml.datasets`: Dataset Functions
-----------------------------------------
.. currentmodule:: openml.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

    attributes_arff_from_df
    check_datasets_active
    create_dataset
    get_dataset
    get_datasets
    list_datasets
    status_update

:mod:`openml.evaluations`: Evaluation Functions
-----------------------------------------------
.. currentmodule:: openml.evaluations

.. autosummary::
   :toctree: generated/
   :template: function.rst

    list_evaluations

:mod:`openml.flows`: Flow Functions
-----------------------------------
.. currentmodule:: openml.flows

.. autosummary::
   :toctree: generated/
   :template: function.rst

    flow_exists
    get_flow
    list_flows
    assert_flows_equal

:mod:`openml.runs`: Run Functions
----------------------------------
.. currentmodule:: openml.runs

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_run
    get_runs
    get_run_trace
    initialize_model_from_run
    initialize_model_from_trace
    list_runs
    run_model_on_task
    run_flow_on_task

:mod:`openml.setups`: Setup Functions
-------------------------------------
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
.. currentmodule:: openml.study

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_study

:mod:`openml.tasks`: Task Functions
-----------------------------------
.. currentmodule:: openml.tasks

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_task
    get_tasks
    list_tasks
