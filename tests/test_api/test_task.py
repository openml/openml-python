# License: BSD 3-Clause
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from requests import Response, Session

import openml
from openml._api import TaskV1API, TaskV2API
from openml.exceptions import OpenMLNotSupportedError
from openml.tasks import OpenMLClassificationTask, OpenMLSplit, TaskType
from openml.tasks.functions import _get_estimation_procedure_list

CLASSIFICATION_TASK_ID = 1
REGRESSION_TASK_ID = 613
LEARNING_CURVE_TASK_ID = 763
DATASTREAM_TASK_ID = 967


@pytest.fixture
def task_v1(http_client_v1, s3_client) -> TaskV1API:
    return TaskV1API(http=http_client_v1, s3=s3_client)


@pytest.fixture
def task_v2(http_client_v2, s3_client) -> TaskV2API:
    return TaskV2API(http=http_client_v2, s3=s3_client)


def _assert_task_1(task):
    assert isinstance(task, OpenMLClassificationTask)
    assert task.task_id == CLASSIFICATION_TASK_ID
    assert task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION
    assert task.dataset_id == 1
    assert task.target_name == "class"
    assert task.estimation_procedure_id == 1
    assert task.estimation_procedure["type"] == "crossvalidation"
    assert task.estimation_procedure["parameters"]["number_folds"] == "10"
    assert task.estimation_procedure["data_splits_url"].endswith("Task_1_splits.arff")


@pytest.mark.test_server
def test_v1_get(task_v1):
    _assert_task_1(task_v1.get(CLASSIFICATION_TASK_ID))


@pytest.mark.test_server
def test_v1_list(task_v1):
    tasks_df = task_v1.list(
        limit=5,
        offset=0,
        task_type=TaskType.SUPERVISED_REGRESSION,
        estimation_procedures=_get_estimation_procedure_list(),
    )
    assert isinstance(tasks_df, pd.DataFrame)
    assert len(tasks_df) == 5
    assert (tasks_df["ttid"] == TaskType.SUPERVISED_REGRESSION).all()
    names = {procedure["name"] for procedure in _get_estimation_procedure_list()}
    assert set(tasks_df["estimation_procedure"]) <= names


@pytest.mark.test_server
def test_v1_download_split(task_v1):
    url = task_v1.get(CLASSIFICATION_TASK_ID).estimation_procedure["data_splits_url"]
    path = task_v1.download_split(url)
    assert path.parts[-4:] == ("get", "1", "Task_1_splits.arff", "body.arff")
    assert isinstance(OpenMLSplit._from_arff_file(path), OpenMLSplit)


@pytest.mark.test_server
def test_v1_download_split_refresh_cache(task_v1):
    url = task_v1.get(CLASSIFICATION_TASK_ID).estimation_procedure["data_splits_url"]
    path = task_v1.download_split(url)
    original = path.read_bytes()
    path.write_bytes(b"@relation stale")

    assert task_v1.download_split(url).read_bytes() == b"@relation stale"
    assert task_v1.download_split(url, refresh_cache=True).read_bytes() == original


@pytest.mark.test_server
def test_v2_get(task_v2):
    _assert_task_1(task_v2.get(CLASSIFICATION_TASK_ID))


def test_v2_list(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.list(limit=5, offset=0, estimation_procedures=[])


@pytest.mark.test_server
def test_v2_download_split(task_v2):
    url = task_v2.get(CLASSIFICATION_TASK_ID).estimation_procedure["data_splits_url"]
    path = task_v2.download_split(url)
    assert path.parts[-4:] == ("get", "1", "Task_1_splits.arff", "body.arff")
    assert isinstance(OpenMLSplit._from_arff_file(path), OpenMLSplit)


@pytest.mark.test_server
def test_v2_evaluation_measure(task_v2):
    task_json = task_v2._http.get(f"tasks/{CLASSIFICATION_TASK_ID}").json()
    inputs = {i["name"]: i for i in task_json["input"]}
    measures = inputs["evaluation_measures"]["evaluation_measures"]
    assert measures["evaluation_measure"] == []
    assert task_v2._create_task_from_json(task_json).evaluation_measure is None

    measures["evaluation_measure"] = "predictive_accuracy"
    task = task_v2._create_task_from_json(task_json)
    assert task.evaluation_measure == "predictive_accuracy"


@pytest.mark.test_server
@pytest.mark.parametrize(
    "task_id",
    [CLASSIFICATION_TASK_ID, REGRESSION_TASK_ID, LEARNING_CURVE_TASK_ID],
)
def test_get_matches(task_v1, task_v2, task_id):
    assert vars(task_v1.get(task_id)) == vars(task_v2.get(task_id))


@pytest.mark.test_server
def test_get_unsupported_task_type_matches(task_v1, task_v2):
    match = "Supervised Data Stream Classification not supported"
    with pytest.raises(NotImplementedError, match=match):
        task_v1.get(DATASTREAM_TASK_ID)
    with pytest.raises(NotImplementedError, match=match):
        task_v2.get(DATASTREAM_TASK_ID)


@pytest.mark.test_server
def test_download_split_matches(task_v1, task_v2):
    url_v1 = task_v1.get(CLASSIFICATION_TASK_ID).estimation_procedure["data_splits_url"]
    url_v2 = task_v2.get(CLASSIFICATION_TASK_ID).estimation_procedure["data_splits_url"]
    assert url_v1 == url_v2
    assert task_v1.download_split(url_v1) == task_v2.download_split(url_v2)


def test_v1_publish(task_v1):
    resource_name = task_v1.resource_type.value
    resource_files = {"description": "Resource Description File"}
    resource_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:upload_task xmlns:oml="http://openml.org/openml">\n'
            f"\t<oml:id>{resource_id}</oml:id>\n"
            f"</oml:upload_task>\n"
        ).encode()

        published_resource_id = task_v1.publish(
            resource_name,
            files=resource_files,
        )

        assert resource_id == published_resource_id

        mock_request.assert_called_once_with(
            method="POST",
            url=openml.config.server + resource_name,
            params={},
            data={"api_key": openml.config.apikey},
            headers=openml.config._HEADERS,
            files=resource_files,
        )


def test_v1_delete(task_v1):
    resource_name = task_v1.resource_type.value
    resource_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_delete xmlns:oml="http://openml.org/openml">\n'
            f"  <oml:id>{resource_id}</oml:id>\n"
            f"</oml:task_delete>\n"
        ).encode()

        task_v1.delete(resource_id)

        mock_request.assert_called_once_with(
            method="DELETE",
            url=(openml.config.server + resource_name + "/" + str(resource_id)),
            params={"api_key": openml.config.apikey},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_tag(task_v1):
    resource_id = 123
    resource_tag = "TAG"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_tag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{resource_id}</oml:id>"
            f"<oml:tag>{resource_tag}</oml:tag>"
            f"</oml:task_tag>"
        ).encode()

        tags = task_v1.tag(resource_id, resource_tag)

        assert resource_tag in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=(openml.config.server + task_v1.resource_type.value + "/tag"),
            params={},
            data={
                "api_key": openml.config.apikey,
                "task_id": resource_id,
                "tag": resource_tag,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_untag(task_v1):
    resource_id = 123
    resource_tag = "TAG"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_untag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{resource_id}</oml:id>"
            f"</oml:task_untag>"
        ).encode()

        tags = task_v1.untag(resource_id, resource_tag)

        assert resource_tag not in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=(openml.config.server + task_v1.resource_type.value + "/untag"),
            params={},
            data={
                "api_key": openml.config.apikey,
                "task_id": resource_id,
                "tag": resource_tag,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v2_publish(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.publish(path=None, files=None)


def test_v2_delete(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.delete(resource_id=None)


def test_v2_tag(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.tag(resource_id=None, tag=None)


def test_v2_untag(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.untag(resource_id=None, tag=None)
