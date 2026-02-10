from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response
    from traitlets import Any

    from openml.tasks.task import OpenMLTask, TaskType


class DatasetAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.DATASET


class TaskAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.TASK

    @abstractmethod
    def get(
        self,
        task_id: int,
    ) -> OpenMLTask:
        """
        API v1:
            GET /task/{task_id}

        API v2:
            GET /tasks/{task_id}
        """
        ...

    # Task listing (V1 only)
    @abstractmethod
    def list(
        self,
        limit: int,
        offset: int,
        task_type: TaskType | int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        List tasks with filters.

        API v1:
            GET /task/list

        API v2:
            Not available.

        Returns
        -------
        pandas.DataFrame
        """
        ...

    def download(
        self,
        url: str,
        handler: Callable[[Response, Path, str], Path] | None = None,
        encoding: str = "utf-8",
        file_name: str = "response.txt",
        md5_checksum: str | None = None,
    ) -> Path:
        return self._http.download(
            url=url,
            handler=handler,
            encoding=encoding,
            file_name=file_name,
            md5_checksum=md5_checksum,
        )


class EvaluationMeasureAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE


class EstimationProcedureAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE


class EvaluationAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.EVALUATION


class FlowAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.FLOW


class StudyAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.STUDY


class RunAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.RUN


class SetupAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.SETUP
