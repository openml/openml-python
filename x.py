# import pytest
# import openml
# from openml.tasks.task import OpenMLTask, TaskType
# from openml._api.resources.tasks import TasksV1, TasksV2


# # ---------- shared helpers ----------

# TEST_TASK_ID = 1          # stable, public task
# TEST_CLASSIF_TASK_ID = 1 # supervised classification
# TEST_TASK_TYPE_ID = 1    # supervised classification


# def assert_basic_task(task: OpenMLTask):
#     assert isinstance(task, OpenMLTask)
#     assert isinstance(task.task_id, int)
#     assert task.task_id > 0
#     assert task.dataset_id is not None
#     assert task.task_type_id in TaskType


# # ---------- V1 tests ----------

# def test_v1_get_task():
#     api = TasksV1(openml.config.get_api_context())

#     task = api.get(TEST_TASK_ID)
#     assert_basic_task(task)


# def test_v1_get_task_with_splits():
#     api = TasksV1(openml.config.get_api_context())

#     task = api.get(TEST_CLASSIF_TASK_ID, download_splits=True)
#     assert_basic_task(task)

#     # only supervised tasks have splits
#     if hasattr(task, "data_splits"):
#         assert task.data_splits is not None


# def test_v1_list_tasks():
#     api = TasksV1(openml.config.get_api_context())

#     df = api.list_tasks(size=5)
#     assert not df.empty
#     assert "tid" in df.columns


# def test_v1_list_tasks_filtered_by_type():
#     api = TasksV1(openml.config.get_api_context())

#     df = api.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION, size=5)
#     assert not df.empty
#     assert all(df["ttid"] == TaskType.SUPERVISED_CLASSIFICATION)


# def test_v1_get_multiple_tasks():
#     api = TasksV1(openml.config.get_api_context())

#     tasks = api.get_tasks([1, 2])
#     assert len(tasks) == 2
#     for t in tasks:
#         assert_basic_task(t)


# # ---------- V2 tests ----------

# def test_v2_get_task():
#     api = TasksV2(openml.config.get_api_context())

#     task = api.get(TEST_TASK_ID)
#     assert_basic_task(task)


# def test_v2_get_task_warns_on_splits():
#     api = TasksV2(openml.config.get_api_context())

#     with pytest.warns(UserWarning):
#         task = api.get(TEST_TASK_ID, download_splits=True)
#         assert_basic_task(task)


# def test_v2_list_task_types():
#     api = TasksV2(openml.config.get_api_context())

#     task_types = api.list_task_types()
#     assert isinstance(task_types, list)
#     assert len(task_types) > 0

#     first = task_types[0]
#     assert "id" in first
#     assert "name" in first


# def test_v2_get_task_type():
#     api = TasksV2(openml.config.get_api_context())

#     tt = api.get_task_type(TEST_TASK_TYPE_ID)
#     assert tt["id"] == TEST_TASK_TYPE_ID
#     assert "name" in tt
#     assert "inputs" in tt
#     assert isinstance(tt["inputs"], list)


# # ---------- cross-version consistency ----------

# def test_v1_v2_same_task_id_consistency():
#     ctx = openml.config.get_api_context()
#     v1 = TasksV1(ctx)
#     v2 = TasksV2(ctx)

#     t1 = v1.get(TEST_TASK_ID)
#     t2 = v2.get(TEST_TASK_ID)

#     assert t1.task_id == t2.task_id
#     assert t1.dataset_id == t2.dataset_id
#     assert t1.task_type_id == t2.task_type_id

import openml
from pprint import pprint
from openml._api.config import settings, APIConfig
from openml._api.http.client import HTTPClient
from openml._api.resources import (
    DatasetsV1,
    DatasetsV2,
    TasksV1,
    TasksV2,
)
from openml._api.resources.tasks import TasksV1, TasksV2
from openml.tasks.task import TaskType


def main():
    v1=APIConfig(
            server="https://www.openml.org/",
            base_url="api/v1/xml/",
            key="...",
        )

    v2=APIConfig(
            server="http://127.0.0.1:8001/",
            base_url="",
            key="...",
        )
    v1_http = HTTPClient(config=settings.api.v1)
    v2_http = HTTPClient(config=settings.api.v2)
    tasks_v1 = TasksV1()
    tasks_v2 = TasksV2()

    TASK_ID = 2
    TASK_TYPE_ID = 1  # Supervised Classification

    print("\n" + "=" * 80)
    print("V1: get(task_id)")
    print("=" * 80)
    t1 = tasks_v1.get(TASK_ID)
    pprint(t1)
    print("type:", type(t1))

    print("\n" + "=" * 80)
    print("V2: get(task_id)")
    print("=" * 80)
    t2 = tasks_v2.get(TASK_ID)
    pprint(t2)
    print("type:", type(t2))

    print("\n" + "=" * 80)
    print("V1: list_tasks(task_type=SUPERVISED_CLASSIFICATION)")
    print("=" * 80)
    df_v1 = tasks_v1.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION, size=5)
    print(df_v1)
    print("shape:", df_v1.shape)

    print("\n" + "=" * 80)
    print("V2: list_task_types()")
    print("=" * 80)
    tt_list = tasks_v2.list_task_types()
    pprint(tt_list)

    print("\n" + "=" * 80)
    print("V2: get_task_type(task_type_id)")
    print("=" * 80)
    tt = tasks_v2.get_task_type(TASK_TYPE_ID)
    pprint(tt)


if __name__ == "__main__":
    main()
