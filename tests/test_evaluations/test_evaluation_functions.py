# License: BSD 3-Clause

import openml
import openml.evaluations
from openml.testing import TestBase


class TestEvaluationFunctions(TestBase):
    _multiprocess_can_split_ = True

    def _check_list_evaluation_setups(self, **kwargs):
        evals_setups = openml.evaluations.list_evaluations_setups("predictive_accuracy",
                                                                  **kwargs,
                                                                  sort_order='desc',
                                                                  output_format='dataframe')
        evals = openml.evaluations.list_evaluations("predictive_accuracy",
                                                    **kwargs,
                                                    sort_order='desc',
                                                    output_format='dataframe')

        # Check if list is non-empty
        self.assertGreater(len(evals_setups), 0)
        # Check if length is accurate
        self.assertEqual(len(evals_setups), len(evals))
        # Check if output from sort is sorted in the right order
        self.assertSequenceEqual(sorted(evals_setups['value'].tolist(), reverse=True),
                                 evals_setups['value'].tolist())

        # Check if output and order of list_evaluations is preserved
        self.assertSequenceEqual(evals_setups['run_id'].tolist(), evals['run_id'].tolist())
        # Check if the hyper-parameter column is as accurate and flow_id
        for index, row in evals_setups.iterrows():
            params = openml.runs.get_run(row['run_id']).parameter_settings
            list1 = [param['oml:value'] for param in params]
            list2 = list(row['parameters'].values())
            # check if all values are equal
            self.assertSequenceEqual(sorted(list1), sorted(list2))
        return evals_setups

    def test_evaluation_list_filter_task(self):
        openml.config.server = self.production_server

        task_id = 7312

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy",
                                                          task=[task_id])

        self.assertGreater(len(evaluations), 100)
        for run_id in evaluations.keys():
            self.assertEqual(evaluations[run_id].task_id, task_id)
            # default behaviour of this method: return aggregated results (not
            # per fold)
            self.assertIsNotNone(evaluations[run_id].value)
            self.assertIsNone(evaluations[run_id].values)

    def test_evaluation_list_filter_uploader_ID_16(self):
        openml.config.server = self.production_server

        uploader_id = 16
        evaluations = openml.evaluations.list_evaluations("predictive_accuracy",
                                                          uploader=[uploader_id],
                                                          output_format='dataframe')
        self.assertEqual(evaluations['uploader'].unique(), [uploader_id])

        self.assertGreater(len(evaluations), 50)

    def test_evaluation_list_filter_uploader_ID_10(self):
        openml.config.server = self.production_server

        setup_id = 10
        evaluations = openml.evaluations.list_evaluations("predictive_accuracy",
                                                          setup=[setup_id])

        self.assertGreater(len(evaluations), 50)
        for run_id in evaluations.keys():
            self.assertEqual(evaluations[run_id].setup_id, setup_id)
            # default behaviour of this method: return aggregated results (not
            # per fold)
            self.assertIsNotNone(evaluations[run_id].value)
            self.assertIsNone(evaluations[run_id].values)

    def test_evaluation_list_filter_flow(self):
        openml.config.server = self.production_server

        flow_id = 100

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy",
                                                          flow=[flow_id])

        self.assertGreater(len(evaluations), 2)
        for run_id in evaluations.keys():
            self.assertEqual(evaluations[run_id].flow_id, flow_id)
            # default behaviour of this method: return aggregated results (not
            # per fold)
            self.assertIsNotNone(evaluations[run_id].value)
            self.assertIsNone(evaluations[run_id].values)

    def test_evaluation_list_filter_run(self):
        openml.config.server = self.production_server

        run_id = 12

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy",
                                                          run=[run_id])

        self.assertEqual(len(evaluations), 1)
        for run_id in evaluations.keys():
            self.assertEqual(evaluations[run_id].run_id, run_id)
            # default behaviour of this method: return aggregated results (not
            # per fold)
            self.assertIsNotNone(evaluations[run_id].value)
            self.assertIsNone(evaluations[run_id].values)

    def test_evaluation_list_limit(self):
        openml.config.server = self.production_server

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy",
                                                          size=100, offset=100)
        self.assertEqual(len(evaluations), 100)

    def test_list_evaluations_empty(self):
        evaluations = openml.evaluations.list_evaluations('unexisting_measure')
        if len(evaluations) > 0:
            raise ValueError('UnitTest Outdated, got somehow results')

        self.assertIsInstance(evaluations, dict)

    def test_evaluation_list_per_fold(self):
        openml.config.server = self.production_server
        size = 1000
        task_ids = [6]
        uploader_ids = [1]
        flow_ids = [6969]

        evaluations = openml.evaluations.list_evaluations(
            "predictive_accuracy", size=size, offset=0, task=task_ids,
            flow=flow_ids, uploader=uploader_ids, per_fold=True)

        self.assertEqual(len(evaluations), size)
        for run_id in evaluations.keys():
            self.assertIsNone(evaluations[run_id].value)
            self.assertIsNotNone(evaluations[run_id].values)
            # potentially we could also test array values, but these might be
            # added in the future

        evaluations = openml.evaluations.list_evaluations(
            "predictive_accuracy", size=size, offset=0, task=task_ids,
            flow=flow_ids, uploader=uploader_ids, per_fold=False)
        for run_id in evaluations.keys():
            self.assertIsNotNone(evaluations[run_id].value)
            self.assertIsNone(evaluations[run_id].values)

    def test_evaluation_list_sort(self):
        openml.config.server = self.production_server
        size = 10
        task_id = 6
        # Get all evaluations of the task
        unsorted_eval = openml.evaluations.list_evaluations(
            "predictive_accuracy", offset=0, task=[task_id])
        # Get top 10 evaluations of the same task
        sorted_eval = openml.evaluations.list_evaluations(
            "predictive_accuracy", size=size, offset=0, task=[task_id], sort_order="desc")
        self.assertEqual(len(sorted_eval), size)
        self.assertGreater(len(unsorted_eval), 0)
        sorted_output = [evaluation.value for evaluation in sorted_eval.values()]
        unsorted_output = [evaluation.value for evaluation in unsorted_eval.values()]

        # Check if output from sort is sorted in the right order
        self.assertTrue(sorted(sorted_output, reverse=True) == sorted_output)

        # Compare manual sorting against sorted output
        test_output = sorted(unsorted_output, reverse=True)
        self.assertTrue(test_output[:size] == sorted_output)

    def test_list_evaluation_measures(self):
        measures = openml.evaluations.list_evaluation_measures()
        self.assertEqual(isinstance(measures, list), True)
        self.assertEqual(all([isinstance(s, str) for s in measures]), True)

    def test_list_evaluations_setups_filter_flow(self):
        openml.config.server = self.production_server
        flow_id = [405]
        size = 100
        evals = self._check_list_evaluation_setups(flow=flow_id, size=size)
        # check if parameters in separate columns works
        evals_cols = openml.evaluations.list_evaluations_setups("predictive_accuracy",
                                                                flow=flow_id, size=size,
                                                                sort_order='desc',
                                                                output_format='dataframe',
                                                                parameters_in_separate_columns=True
                                                                )
        columns = (list(evals_cols.columns))
        keys = (list(evals['parameters'].values[0].keys()))
        self.assertTrue(all(elem in columns for elem in keys))

    def test_list_evaluations_setups_filter_task(self):
        openml.config.server = self.production_server
        task_id = [6]
        size = 121
        self._check_list_evaluation_setups(task=task_id, size=size)
