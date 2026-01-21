import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd

from openml._api.resources.tasks import TasksV1, TasksV2
from openml.tasks import (
    TaskType,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
    list_tasks,
    get_task,
    get_tasks,
    delete_task,
    create_task
)

class TestTasksEndpoints(unittest.TestCase):

    def setUp(self):
        # We mock the HTTP client (requests session) used by the API classes
        self.mock_http = MagicMock()

    def test_v1_get_endpoint(self):
        """Test GET task/{id} endpoint construction and parsing"""
        api = TasksV1(self.mock_http)
        
        # We include two parameters to ensure xmltodict parses 'oml:parameter' 
        # as a list, preventing the TypeError seen previously.
        self.mock_http.get.return_value.text = """
        <oml:task xmlns:oml="http://openml.org/openml">
            <oml:task_id>1</oml:task_id>
            <oml:task_type_id>1</oml:task_type_id>
            <oml:task_type>Supervised Classification</oml:task_type>
            <oml:input name="source_data">
                <oml:data_set>
                    <oml:data_set_id>100</oml:data_set_id>
                    <oml:target_feature>class</oml:target_feature>
                </oml:data_set>
            </oml:input>
            <oml:input name="estimation_procedure">
                <oml:estimation_procedure>
                    <oml:id>1</oml:id>
                    <oml:type>crossvalidation</oml:type>
                    <oml:data_splits_url>http://splits</oml:data_splits_url>
                    <oml:parameter name="folds">10</oml:parameter>
                    <oml:parameter name="stratified">true</oml:parameter>
                </oml:estimation_procedure>
            </oml:input>
        </oml:task>
        """

        task = api.get(1)

        self.mock_http.get.assert_called_with("task/1")
        self.assertIsInstance(task, OpenMLClassificationTask)
        self.assertEqual(task.task_id, 1)

    def test_v1_list_endpoint_url_construction(self):
        """Test list tasks endpoint URL generation with filters"""
        api = TasksV1(self.mock_http)
        
        # We mock `_fetch_tasks_df` because parsing the list XML is complex 
        # and we just want to verify the URL parameters here.
        with patch.object(api, '_fetch_tasks_df') as mock_fetch:
            api.list(
                limit=100,
                offset=50,
                task_type=TaskType.SUPERVISED_CLASSIFICATION,
                tag="study_14"
            )
            
            # Verify the constructed API call string passed to the fetcher
            expected_call = "task/list/limit/100/offset/50/type/1/tag/study_14"
            mock_fetch.assert_called_with(api_call=expected_call)


    def test_v2_get_endpoint(self):
        """Test GET tasks/{id} V2 endpoint"""
        api = TasksV2(self.mock_http)

        # JSON response structure matches what V2 expects
        self.mock_http.get.return_value.json.return_value = {
            "id": 500,
            "task_type_id": "2",  # Regression
            "task_type": "Supervised Regression",
            "input": [
                {
                    "name": "source_data",
                    "data_set": {"data_set_id": "99", "target_feature": "price"}
                },
                {
                    "name": "estimation_procedure",
                    "estimation_procedure": {
                        "id": "5",
                        "type": "cv",
                        "parameter": []
                    }
                }
            ]
        }

        task = api.get(500)

        self.mock_http.get.assert_called_with("tasks/500")
        self.assertIsInstance(task, OpenMLRegressionTask)
        self.assertEqual(task.target_name, "price")

    def test_v2_list_not_available(self):
        """Ensure V2 list endpoint raises error (as per code)"""
        api = TasksV2(self.mock_http)
        with self.assertRaises(NotImplementedError):
            api.list(limit=10, offset=0)


class TestTaskHighLevelFunctions(unittest.TestCase):
    """Test the user-facing functions in functions.py"""

    @patch("openml.tasks.functions.api_context")
    def test_list_tasks_wrapper(self, mock_api_context):
        """Test list_tasks() calls the backend correctly"""
        # Setup backend to return a dummy dataframe
        mock_api_context.backend.tasks.list.return_value = pd.DataFrame({'id': [1]})
        
        list_tasks(
            task_type=TaskType.SUPERVISED_CLASSIFICATION,
            offset=10,
            size=50,
            tag="my_tag"
        )
        
        # The backend list method is called with positional arguments for limit (size) 
        # and offset because of how `_list_all` works internally.
        mock_api_context.backend.tasks.list.assert_called_with(
            50,  # limit (size)
            10,  # offset
            task_type=TaskType.SUPERVISED_CLASSIFICATION,
            tag="my_tag",
            data_tag=None,
            status=None,
            data_id=None,
            data_name=None,
            number_instances=None,
            number_features=None,
            number_classes=None,
            number_missing_values=None
        )

    @patch("openml.tasks.functions.get_dataset")
    @patch("openml.tasks.functions.api_context")
    def test_get_task_wrapper(self, mock_api_context, mock_get_dataset):
        """Test get_task() retrieves task and dataset"""
        # Mock Task
        mock_task_obj = MagicMock()
        mock_task_obj.dataset_id = 123
        mock_task_obj.target_name = "class"
        mock_api_context.backend.tasks.get.return_value = mock_task_obj
        
        # Mock Dataset (needed for class labels)
        mock_dataset = MagicMock()
        mock_get_dataset.return_value = mock_dataset

        get_task(task_id=10, download_data=False)

        # Verify calls
        mock_api_context.backend.tasks.get.assert_called_with(10)
        
        # `get_task` passes kwargs directly to get_dataset.
        mock_get_dataset.assert_called_with(123, download_data=False)

    @patch("openml.tasks.functions.get_task")
    def test_get_tasks_list_wrapper(self, mock_get_task):
        """Test get_tasks() iterates and calls get_task() for each ID"""
        ids_to_fetch = [100, 101]
        
        # Execute the bulk fetch
        get_tasks(ids_to_fetch, download_data=False, download_qualities=False)
        
        # Verify `get_task` was called exactly twice
        self.assertEqual(mock_get_task.call_count, 2)
        
        # Verify the arguments for each call
        expected_calls = [
            call(100, download_data=False, download_qualities=False),
            call(101, download_data=False, download_qualities=False)
        ]
        mock_get_task.assert_has_calls(expected_calls)

    @patch("openml.utils._delete_entity")
    def test_delete_task_wrapper(self, mock_delete):
        """Test delete_task() hits the delete endpoint"""
        delete_task(999)
        mock_delete.assert_called_with("task", 999)

    def test_create_task_factory(self):
        """Test create_task() returns correct object (no API call until publish)"""
        task = create_task(
            task_type=TaskType.SUPERVISED_CLASSIFICATION,
            dataset_id=1,
            estimation_procedure_id=1,
            target_name="class"
        )
        self.assertIsInstance(task, OpenMLClassificationTask)
        self.assertEqual(task.dataset_id, 1)