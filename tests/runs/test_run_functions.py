import openml
from openml.testing import TestBase


class TestRun(TestBase):
    def test_get_run(self):
        run = openml.runs.get_run(473350)
        self.assertEqual(run.dataset_id, 1167)
        self.assertEqual(run.evaluations['f_measure'], 0.624668)
        for i, value in [(0, 0.66233),
                         (1, 0.639286),
                         (2, 0.567143),
                         (3, 0.745833),
                         (4, 0.599638),
                         (5, 0.588801),
                         (6, 0.527976),
                         (7, 0.666365),
                         (8, 0.56759),
                         (9, 0.64621)]:
            self.assertEqual(run.detailed_evaluations['f_measure'][0][i], value)

    def _check_run(self, run):
        self.assertIsInstance(run, dict)
        self.assertEqual(len(run), 5)

    def test_get_runs_list(self):
        runs = openml.runs.list_runs(2)
        self.assertEqual(len(runs), 1)
        self._check_run(runs[0])

    def test_get_runs_list_by_task(self):
        runs = openml.runs.list_runs_by_task(1)
        self.assertGreaterEqual(len(runs), 600)
        for run in runs:
            self._check_run(run)
        num_runs = len(runs)

        runs = openml.runs.list_runs_by_task([1, 2])
        self.assertGreaterEqual(len(runs), num_runs + 1)
        for run in runs:
            self._check_run(run)

    def test_get_runs_list_by_uploader(self):
        # 29 is Dominik Kirchhoff - Joaquin and Jan have too many runs right now
        runs = openml.runs.list_runs_by_uploader(29)
        self.assertGreaterEqual(len(runs), 3)
        for run in runs:
            self._check_run(run)
        num_runs = len(runs)

        runs = openml.runs.list_runs_by_uploader([29, 274])
        self.assertGreaterEqual(len(runs), num_runs + 1)
        for run in runs:
            self._check_run(run)

    def test_get_runs_list_by_flow(self):
        runs = openml.runs.list_runs_by_flow(1)
        self.assertGreaterEqual(len(runs), 1)
        for run in runs:
            self._check_run(run)
        num_runs = len(runs)

        runs = openml.runs.list_runs_by_flow([1154, 1069])
        self.assertGreaterEqual(len(runs), num_runs + 1)
        for run in runs:
            self._check_run(run)

    def test_get_runs_list_by_filters(self):
        ids = [505212, 6100]
        tasks = [2974, 339]
        uploaders_1 = [1, 17]
        uploaders_2 = [29, 274]
        flows = [74, 1718]

        self.assertRaises(ValueError, openml.runs.list_runs_by_filters)

        runs = openml.runs.list_runs_by_filters(id=ids)
        self.assertEqual(len(runs), 2)

        runs = openml.runs.list_runs_by_filters(task=tasks)
        self.assertGreaterEqual(len(runs), 2)

        runs = openml.runs.list_runs_by_filters(uploader=uploaders_2)
        self.assertGreaterEqual(len(runs), 10)

        runs = openml.runs.list_runs_by_filters(flow=flows)
        self.assertGreaterEqual(len(runs), 100)

        runs = openml.runs.list_runs_by_filters(id=ids, task=tasks,
                                                uploader=uploaders_1)

    def test_get_runs_list_by_tag(self):
        runs = openml.runs.list_runs_by_tag('02-11-16_21.46.39')
        self.assertEqual(len(runs), 1)