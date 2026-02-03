from time import time
import pytest
from openml.testing import TestAPIBase
from openml._api import ResourceV1API, ResourceV2API, FallbackProxy
from openml.enums import ResourceType
from openml.exceptions import OpenMLNotSupportedError


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
        resource_id = 1
        unique_indicator = str(time()).replace(".", "")
        tag = f"TestResourceV1API_test_tag_and_untag_{unique_indicator}"

        tags = self.resource.tag(resource_id, tag)
        self.assertIn(tag, tags)

        tags = self.resource.untag(resource_id, tag)
        self.assertNotIn(tag, tags)


class TestResourceV2API(TestResourceV1API):
    def setUp(self):
        super().setUp()

        self.server = ""
        self.base_url = ""
        self.api_key = ""
        self.http_client = self._get_http_client(
            server=self.server,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )

        self.resource = ResourceV2API(self.http_client)
        self.resource.resource_type = ResourceType.TASK

    @pytest.mark.xfail(raises=OpenMLNotSupportedError)
    def test_publish_and_delete(self):
        super().test_tag_and_untag()


    @pytest.mark.xfail(raises=OpenMLNotSupportedError)
    def test_tag_and_untag(self):
        super().test_tag_and_untag()


class TestResourceFallbackAPI(TestResourceV1API):
    def setUp(self):
        super().setUp()

        self.http_client_v2 = self._get_http_client(
            server="",
            base_url="",
            api_key="",
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )

        resource_v1 = ResourceV1API(self.http_client)
        resource_v1.resource_type = ResourceType.TASK

        resource_v2 = ResourceV2API(self.http_client_v2)
        resource_v2.resource_type = ResourceType.TASK

        self.resource = FallbackProxy(resource_v2, resource_v1)
