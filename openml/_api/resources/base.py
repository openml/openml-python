from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from build.lib.openml.tasks.task import TaskType
    from requests import Response

    from openml._api.http import HTTPClient
    from openml.datasets.dataset import OpenMLDataset
    from openml.tasks.task import OpenMLTask


class ResourceAPI:
    def __init__(self, http: HTTPClient):
        self._http = http


class DatasetsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(self, dataset_id: int) -> OpenMLDataset | tuple[OpenMLDataset, Response]: ...


class TasksAPI(ResourceAPI, ABC):
    # Single task retrieval (V1 and V2)
    @abstractmethod
    def get(
        self,
        task_id: int,
        download_splits: bool = False, # noqa: FBT001, FBT002
        **get_dataset_kwargs: Any,
    ) -> OpenMLTask:
        """
        API v1:
            GET /task/{task_id}

        API v2:
            GET /tasks/{task_id}
        """
        ...

    # # Multiple task retrieval (V1 only)
    # @abstractmethod
    # def get_tasks(
    #     self,
    #     task_ids: list[int],
    #     **kwargs: Any,
    # ) -> list[OpenMLTask]:
    #     """
    #     Retrieve multiple tasks.

    #     API v1:
    #         Implemented via repeated GET /task/{task_id}

    #     API v2:
    #         Not currently supported

    #     Parameters
    #     ----------
    #     task_ids : list[int]

    #     Returns
    #     -------
    #     list[OpenMLTask]
    #     """
    #     ...

    # # Task listing (V1 only)
    # @abstractmethod
    # def list_tasks(
    #     self,
    #     *,
    #     task_type: TaskType | None = None,
    #     offset: int | None = None,
    #     size: int | None = None,
    #     **filters: Any,
    # ):
    #     """
    #     List tasks with filters.

    #     API v1:
    #         GET /task/list

    #     API v2:
    #         Not available.

    #     Returns
    #     -------
    #     pandas.DataFrame
    #     """
    #     ...

    # # Task creation (V1 only)
    # @abstractmethod
    # def create_task(
    #     self,
    #     task_type: TaskType,
    #     dataset_id: int,
    #     estimation_procedure_id: int,
    #     **kwargs: Any,
    # ) -> OpenMLTask:
    #     """
    #     Create a new task.

    #     API v1:
    #         POST /task

    #     API v2:
    #         Not supported.

    #     Returns
    #     -------
    #     OpenMLTask
    #     """
    #     ...

    # # Task deletion (V1 only)
    # @abstractmethod
    # def delete_task(self, task_id: int) -> bool:
    #     """
    #     Delete a task.

    #     API v1:
    #         DELETE /task/{task_id}

    #     API v2:
    #         Not supported.

    #     Returns
    #     -------
    #     bool
    #     """
    #     ...

    # # Task type listing (V2 only)
    # @abstractmethod
    # def list_task_types(self) -> list[dict[str, Any]]:
    #     """
    #     List all task types.

    #     API v2:
    #         GET /tasktype/list

    #     API v1:
    #         Not available.

    #     Returns
    #     -------
    #     list[dict]
    #     """
    #     ...

    # # Task type retrieval (V2 only)
    # @abstractmethod
    # def get_task_type(self, task_type_id: int) -> dict[str, Any]:
    #     """
    #     Retrieve a single task type.

    #     API v2:
    #         GET /tasktype/{task_type_id}

    #     API v1:
    #         Not available.

    #     Returns
    #     -------
    #     dict
    #     """
    #     ...
