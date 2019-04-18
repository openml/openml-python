import unittest
from random import randint
from time import time

from openml.testing import TestBase
from openml.datasets import (
    check_datasets_active,
    get_dataset,
    list_datasets,
    OpenMLDataset,
)
from openml.tasks import (
    create_task,
    get_task
)
from openml.utils import (
    _delete_entity,
)


class OpenMLTaskTest(TestBase):
    """
    A helper class. The methods of the test case
    are only executed in subclasses of the test case.
    """
    @classmethod
    def setUpClass(cls):
        if cls is OpenMLTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLTaskTest tests,"
                " it's a base class"
            )

    def setUp(self):
        super(OpenMLTaskTest, self).setUp()

    def test_download_task(self):

        return get_task(self.task_id)

    def test_upload_task(self):

        dataset_id = self._get_compatible_rand_dataset()
        # TODO consider implementing on the diff task types.
        task = create_task(
            task_type_id=self.task_type_id,
            dataset_id=dataset_id,
            target_name=self._get_random_feature(dataset_id),
            estimation_procedure_id=self.estimation_procedure
        )

        task_id = task.publish()
        _delete_entity('task', task_id)

    def _get_compatible_rand_dataset(self) -> int:

        compatible_datasets = []
        active_datasets = list_datasets(status='active')

        # depending on the task type, find either datasets
        # with only symbolic features or datasets with only
        # numerical features.
        if self.task_type_id != 2:
            for dataset_id, dataset_info in active_datasets.items():
                # extra checks because of:
                # https://github.com/openml/OpenML/issues/959
                if 'NumberOfNumericFeatures' in dataset_info:
                    if dataset_info['NumberOfNumericFeatures'] == 0:
                        compatible_datasets.append(dataset_id)
        else:
            for dataset_id, dataset_info in active_datasets.items():
                if 'NumberOfSymbolicFeatures' in dataset_info:
                    if dataset_info['NumberOfSymbolicFeatures'] == 0:
                        compatible_datasets.append(dataset_id)

        random_dataset_pos = randint(0, len(compatible_datasets) - 1)

        return compatible_datasets[random_dataset_pos]

    def _get_random_feature(self, dataset_id: int) -> str:

        random_dataset = get_dataset(dataset_id)
        # necessary loop to overcome string and date type
        # features.
        while True:
            random_feature_index = randint(0, len(random_dataset.features) - 1)
            random_feature = random_dataset.features[random_feature_index]
            if self.task_type_id == 2:
                if random_feature.data_type == 'numeric':
                    break
            else:
                if random_feature.data_type == 'nominal':
                    break
        return random_feature.name

    def _reupload_dataset(self, dataset: OpenMLDataset) -> int:
        """Reupload the dataset.

        Add a sentinel to the dataset name to achieve a
        successful upload every time without creating a
        new dataset.

        Parameters
        ----------
        dataset: OpenMLDataset
            The dataset from OpenML that will be
            reuploaded.

        Returns
        -------
        int
            Dataset id. If the reupload is successful,
            the new id. Otherwise, the old id of the
            dataset.
        """
        dataset.name = '%s%s' % (self._get_sentinel(), dataset.name)
        # Providing both dataset file and url
        # raises an error when uploading.
        dataset.url = None

        return dataset.publish()

    @staticmethod
    def _wait_dataset_activation(
            dataset_id: int,
            max_wait_time: int
    ):
        """Wait until the dataset status is changed
        to activated, given a max wait time.

        Parameters
        ----------
        dataset_id: int
            The id of the dataset whose status
            activation will be observed.
        max_wait_time: int
            Maximal amount of time to wait in
            seconds.
        """
        start_time = time()
        # Check while the status of the dataset is not activated
        while not check_datasets_active([dataset_id]).get(dataset_id):
            # break if the time so far exceeds max wait time
            if time() - start_time > max_wait_time:
                break
