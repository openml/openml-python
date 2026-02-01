import pytest
from openml.testing import TestAPIBase
from openml._api.resources.base.versions import ResourceV1API
from openml._api.config import ResourceType


class TestResourceV1API(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.resource = ResourceV1API(self.http_client)
        self.resource.resource_type = ResourceType.TASK

    @pytest.mark.uses_test_server()
    def test_publish_and_delete(self):
        task_xml = """
        <oml:task_inputs xmlns:oml="http://openml.org/openml">
            <oml:task_type_id>5</oml:task_type_id>
            <oml:input name="source_data">193</oml:input>
            <oml:input name="estimation_procedure">17</oml:input>
        </oml:task_inputs>
        """

        task_id = None
        try:
            # Publish the task
            task_id = self.resource.publish(
                "task",
                files={"description": task_xml},
            )

            # Get the task to verify it exists
            get_response = self.http_client.get(f"task/{task_id}")
            self.assertEqual(get_response.status_code, 200)

        finally:
            # delete the task if it was created
            if task_id is not None:
                success = self.resource.delete(task_id)
                self.assertTrue(success)


    @pytest.mark.uses_test_server()
    def test_tag_and_untag(self):
        pass
