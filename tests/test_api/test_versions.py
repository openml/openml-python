import pytest
from requests import Session, Response
from unittest.mock import patch
from openml.testing import TestAPIBase
from openml._api import FallbackProxy, ResourceAPI
from openml.enums import ResourceType, APIVersion
from openml.exceptions import OpenMLNotSupportedError


class TestResourceAPIBase(TestAPIBase):
    resource: ResourceAPI | FallbackProxy

    @property
    def http_client(self):
        return self.resource._http

    def _publish(self):
        resource_name = "task"
        resource_files = {"description": """Resource Description File"""}
        resource_id = 123

        with patch.object(Session, "request") as mock_request:
            mock_request.return_value = Response()
            mock_request.return_value.status_code = 200
            mock_request.return_value._content = f'<oml:upload_task xmlns:oml="http://openml.org/openml">\n\t<oml:id>{resource_id}</oml:id>\n</oml:upload_task>\n'.encode("utf-8")

            published_resource_id = self.resource.publish(
                resource_name,
                files=resource_files,
            )

            self.assertEqual(resource_id, published_resource_id)

            mock_request.assert_called_once_with(
                method="POST",
                url=self.http_client.server + self.http_client.base_url + resource_name,
                params={},
                data={'api_key': self.http_client.api_key},
                headers=self.http_client.headers,
                files=resource_files,
            )

    def _delete(self):
        resource_name = "task"
        resource_id = 123

        with patch.object(Session, "request") as mock_request:
            mock_request.return_value = Response()
            mock_request.return_value.status_code = 200
            mock_request.return_value._content = f'<oml:task_delete xmlns:oml="http://openml.org/openml">\n  <oml:id>{resource_id}</oml:id>\n</oml:task_delete>\n'.encode("utf-8")

            self.resource.delete(resource_id)

            mock_request.assert_called_once_with(
                method="DELETE",
                url=self.http_client.server + self.http_client.base_url + resource_name + "/" + str(resource_id),
                params={'api_key': self.http_client.api_key},
                data={},
                headers=self.http_client.headers,
                files=None,
            )

    def _tag(self):
        resource_id = 123
        resource_tag = "TAG"

        with patch.object(Session, "request") as mock_request:
            mock_request.return_value = Response()
            mock_request.return_value.status_code = 200
            mock_request.return_value._content = f'<oml:task_tag xmlns:oml="http://openml.org/openml"><oml:id>{resource_id}</oml:id><oml:tag>{resource_tag}</oml:tag></oml:task_tag>'.encode("utf-8")

            tags = self.resource.tag(resource_id, resource_tag)
            self.assertIn(resource_tag, tags)

            mock_request.assert_called_once_with(
                method="POST",
                url=self.http_client.server + self.http_client.base_url + self.resource.resource_type + "/tag",
                params={},
                data={'api_key': self.http_client.api_key, 'task_id': resource_id, 'tag': resource_tag},
                headers=self.http_client.headers,
                files=None,
            )

    def _untag(self):
        resource_id = 123
        resource_tag = "TAG"

        with patch.object(Session, "request") as mock_request:
            mock_request.return_value = Response()
            mock_request.return_value.status_code = 200
            mock_request.return_value._content = f'<oml:task_untag xmlns:oml="http://openml.org/openml"><oml:id>{resource_id}</oml:id></oml:task_untag>'.encode("utf-8")

            tags = self.resource.untag(resource_id, resource_tag)
            self.assertNotIn(resource_tag, tags)

            mock_request.assert_called_once_with(
                method="POST",
                url=self.http_client.server + self.http_client.base_url + self.resource.resource_type + "/untag",
                params={},
                data={'api_key': self.http_client.api_key, 'task_id': resource_id, 'tag': resource_tag},
                headers=self.http_client.headers,
                files=None,
            )

class TestResourceV1API(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        self.resource = self._create_resource(
            api_version=APIVersion.V1,
            resource_type=ResourceType.TASK,
        )

    def test_publish(self):
        self._publish()

    def test_delete(self):
        self._delete()

    def test_tag(self):
        self._tag()

    def test_untag(self):
        self._untag()


class TestResourceV2API(TestResourceAPIBase):
    def setUp(self):
        super().setUp()
        self.resource = self._create_resource(
            api_version=APIVersion.V2,
            resource_type=ResourceType.TASK,
        )

    def test_publish(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._publish()

    def test_delete(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._delete()

    def test_tag(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._tag()

    def test_untag(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._untag()


class TestResourceFallbackAPI(TestResourceAPIBase):
    @property
    def http_client(self):
        # since these methods are not implemented for v2, they will fallback to v1 api
        return self.http_clients[APIVersion.V1]

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

    def test_publish(self):
        self._publish()

    def test_delete(self):
        self._delete()

    def test_tag(self):
        self._tag()

    def test_untag(self):
        self._untag()
