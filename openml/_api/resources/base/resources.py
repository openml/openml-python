from __future__ import annotations

from typing import TYPE_CHECKING

from openml._api.resources.base import ResourceAPI
from openml.enums import ResourceType

if TYPE_CHECKING:
    import pandas as pd

    from openml._api.clients import HTTPClient, MinIOClient


class DatasetAPI(ResourceAPI):
    """Abstract API interface for dataset resources."""

    resource_type: ResourceType = ResourceType.DATASET

    def __init__(self, http: HTTPClient, minio: MinIOClient):
        self._minio = minio
        super().__init__(http)


class TaskAPI(ResourceAPI):
    """Abstract API interface for task resources."""

    resource_type: ResourceType = ResourceType.TASK


class EvaluationMeasureAPI(ResourceAPI):
    """Abstract API interface for evaluation measure resources."""

    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE


class EstimationProcedureAPI(ResourceAPI):
    """Abstract API interface for estimation procedure resources."""

    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE


class EvaluationAPI(ResourceAPI):
    """Abstract API interface for evaluation resources."""

    resource_type: ResourceType = ResourceType.EVALUATION


class FlowAPI(ResourceAPI):
    """Abstract API interface for flow resources."""

    resource_type: ResourceType = ResourceType.FLOW


class StudyAPI(ResourceAPI):
    """Abstract API interface for study resources."""

    resource_type: ResourceType = ResourceType.STUDY

    def list(  # noqa: PLR0913
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: str | None = None,
        main_entity_type: str | None = None,
        uploader: list[int] | None = None,
        benchmark_suite: int | None = None,
    ) -> pd.DataFrame:
        """List studies from the OpenML server.

        Parameters
        ----------
        limit : int, optional
            Maximum number of studies to return.
        offset : int, optional
            Number of studies to skip.
        status : str, optional
            Filter by status (active, in_preparation, deactivated, all).
        main_entity_type : str, optional
            Filter by main entity type (run, task).
        uploader : list[int], optional
            Filter by uploader IDs.
        benchmark_suite : int, optional
            Filter by benchmark suite ID.

        Returns
        -------
        pd.DataFrame
            DataFrame containing study information.
        """
        raise NotImplementedError("Subclasses must implement list method")


class RunAPI(ResourceAPI):
    """Abstract API interface for run resources."""

    resource_type: ResourceType = ResourceType.RUN


class SetupAPI(ResourceAPI):
    """Abstract API interface for setup resources."""

    resource_type: ResourceType = ResourceType.SETUP
