from time import time
import pytest
from openml.testing import TestBase, TestAPIBase
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
        # publish
        task_id = self.resource.publish(
            "task",
            files={"description": task_xml},
        )
        self.assertIsNotNone(task_id)

        # cleanup incase of failure
        TestBase._mark_entity_for_removal("task", task_id)
        TestBase.logger.info(f"collected from {__file__}: {task_id}")

        # delete
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
        self.resource = self._create_resource(
            api_version=APIVersion.V1,
            resource_type=ResourceType.TASK,
        )

    def test_publish_and_delete(self):
        self._publish_and_delete()

    def test_tag_and_untag(self):
        self._tag_and_untag()


class TestResourceV2API(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        self.resource = self._create_resource(
            api_version=APIVersion.V2,
            resource_type=ResourceType.TASK,
        )

    def test_publish_and_delete(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._tag_and_untag()

    def test_tag_and_untag(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._tag_and_untag()


class TestResourceFallbackAPI(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        resource_v1 = self._create_resource(
            api_version=APIVersion.V1,
            resource_type=ResourceType.TASK,
        )
        resource_v2 = self._create_resource(
            api_version=APIVersion.V2,
            resource_type=ResourceType.TASK,
        )
        self.resource = FallbackProxy(resource_v2, resource_v1)

    def test_publish_and_delete(self):
        self._publish_and_delete()

    def test_tag_and_untag(self):
        self._tag_and_untag()
