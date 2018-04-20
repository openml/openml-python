from openml.testing import TestBase
import openml


class OpenMLTaskTest(TestBase):
    _multiprocess_can_split_ = True

    def test_list_all(self):
        list_datasets = openml.datasets.functions._list_datasets
        datasets = openml.utils.list_all(list_datasets)

        self.assertGreaterEqual(len(datasets), 100)
        for did in datasets:
            self._check_dataset(datasets[did])

        # TODO implement these tests
        # datasets = openml.utils.list_all(list_datasets, limit=50)
        # self.assertEqual(len(datasets), 50)