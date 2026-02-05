from time import time
import pytest
from openml.testing import TestAPIBase
from openml._api import ResourceV1API, ResourceV2API, FallbackProxy, ResourceAPI
from openml.enums import ResourceType, APIVersion
from openml.exceptions import OpenMLNotSupportedError


@pytest.mark.uses_test_server()
class TestResourceAPIBase(TestAPIBase):
    resource: ResourceAPI | FallbackProxy

    def _publish_and_delete(self):
        task_xml = """
        <oml:task_inputs xmlns:oml="http://openml.org/openml">
            <oml:task_type_id>5</oml:task_type_id>
            <oml:input name="source_data">193</oml:input>
            <oml:input name="estimation_procedure">17</oml:input>
        </oml:task_inputs>
        """

        task_id = self.resource.publish(
            "task",
            files={"description": task_xml},
        )
        self.assertIsNotNone(task_id)

        success = self.resource.delete(task_id)
        self.assertTrue(success)

    def _tag_and_untag(self):
        resource_id = 1
        unique_indicator = str(time()).replace(".", "")
        tag = f"{self.__class__.__name__}_test_tag_and_untag_{unique_indicator}"

        tags = self.resource.tag(resource_id, tag)
        self.assertIn(tag, tags)

        tags = self.resource.untag(resource_id, tag)
        self.assertNotIn(tag, tags)


class TestResourceV1API(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V1]
        self.resource = ResourceV1API(http_client)
        self.resource.resource_type = ResourceType.TASK

    def test_publish_and_delete(self):
        self._publish_and_delete()

    def test_tag_and_untag(self):
        self._tag_and_untag()


class TestResourceV2API(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V2]
        self.resource = ResourceV2API(http_client)
        self.resource.resource_type = ResourceType.TASK

    def test_publish_and_delete(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._tag_and_untag()

    def test_tag_and_untag(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._tag_and_untag()


class TestResourceFallbackAPI(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        http_client_v1 = self.http_clients[APIVersion.V1]
        resource_v1 = ResourceV1API(http_client_v1)
        resource_v1.resource_type = ResourceType.TASK

        http_client_v2 = self.http_clients[APIVersion.V2]
        resource_v2 = ResourceV2API(http_client_v2)
        resource_v2.resource_type = ResourceType.TASK

        self.resource = FallbackProxy(resource_v2, resource_v1)

    def test_publish_and_delete(self):
        self._publish_and_delete()

    def test_tag_and_untag(self):
        self._tag_and_untag()
