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


:mod:`openml.datasets`: Dataset Functions
-----------------------------------------
.. currentmodule:: openml.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

    check_datasets_active
    get_dataset
    get_datasets
    list_datasets

:mod:`openml.runs`: Run Functions
----------------------------------
.. currentmodule:: openml.runs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   run_task
   get_run
   get_runs
   list_runs
   list_runs_by_flow
   list_runs_by_tag
   list_runs_by_task
   list_runs_by_uploader
   list_runs_by_filters
   run_model_on_task
   run_flow_on_task
   get_run_trace
   initialize_model_from_run
   initialize_model_from_trace

:mod:`openml.tasks`: Task Functions
-----------------------------------
.. currentmodule:: openml.tasks

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_task
    get_tasks
    list_tasks

:mod:`openml.flows`: Flow Functions
-----------------------------------
.. currentmodule:: openml.flow

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_flow
    list_flows
    flow_exists
 
:mod:`openml.flows`: Evaluation Functions
-----------------------------------------
.. currentmodule:: openml.evaluation

.. autosummary::
   :toctree: generated/
   :template: function.rst

    list_evaluations
 
