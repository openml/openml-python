from __future__ import annotations

import builtins
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    import pandas as pd

    from openml.datasets.data_feature import OpenMLDataFeature
    from openml.datasets.dataset import OpenMLDataset
    from openml.estimation_procedures import OpenMLEstimationProcedure
    from openml.evaluations import OpenMLEvaluation
    from openml.flows.flow import OpenMLFlow
    from openml.runs.run import OpenMLRun
    from openml.setups.setup import OpenMLSetup
    from openml.tasks.task import TaskType


class DatasetAPI(ResourceAPI):
    """Abstract API interface for dataset resources."""

    resource_type: ResourceType = ResourceType.DATASET

    @abstractmethod
    def get(  # noqa: PLR0913
        self,
        dataset_id: int,
        download_data: bool = False,  # noqa: FBT002
        cache_format: Literal["pickle", "feather"] = "pickle",
        download_qualities: bool = False,  # noqa: FBT002
        download_features_meta_data: bool = False,  # noqa: FBT002
        download_all_files: bool = False,  # noqa: FBT002
        force_refresh_cache: bool = False,  # noqa: FBT002
    ) -> OpenMLDataset: ...

    @abstractmethod
    def list(
        self,
        limit: int,
        offset: int,
        *,
        data_id: list[int] | None = None,  # type: ignore
        **kwargs: Any,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def edit(  # noqa: PLR0913
        self,
        dataset_id: int,
        description: str | None = None,
        creator: str | None = None,
        contributor: str | None = None,
        collection_date: str | None = None,
        language: str | None = None,
        default_target_attribute: str | None = None,
        ignore_attribute: str | list[str] | None = None,  # type: ignore
        citation: str | None = None,
        row_id_attribute: str | None = None,
        original_data_url: str | None = None,
        paper_url: str | None = None,
    ) -> int: ...

    @abstractmethod
    def fork(self, dataset_id: int) -> int: ...

    @abstractmethod
    def status_update(self, dataset_id: int, status: Literal["active", "deactivated"]) -> None: ...

    @abstractmethod
    def list_qualities(self) -> builtins.list[str]: ...

    @abstractmethod
    def feature_add_ontology(self, dataset_id: int, index: int, ontology: str) -> bool: ...

    @abstractmethod
    def feature_remove_ontology(self, dataset_id: int, index: int, ontology: str) -> bool: ...

    @abstractmethod
    def get_features(
        self,
        dataset_id: int,
        force_refresh_cache: bool = False,  # noqa: FBT002
    ) -> dict[int, OpenMLDataFeature]: ...

    @abstractmethod
    def get_qualities(
        self,
        dataset_id: int,
        force_refresh_cache: bool = False,  # noqa: FBT002
    ) -> dict[str, float] | None: ...

    @abstractmethod
    def parse_features_file(
        self, features_file: Path, features_pickle_file: Path | None = None
    ) -> dict[int, OpenMLDataFeature]: ...

    @abstractmethod
    def parse_qualities_file(
        self, qualities_file: Path, qualities_pickle_file: Path | None = None
    ) -> dict[str, float]: ...

    @abstractmethod
    def _download_file(self, url_ext: str) -> Path: ...

    @abstractmethod
    def download_features_file(self, dataset_id: int) -> Path: ...

    @abstractmethod
    def download_qualities_file(self, dataset_id: int) -> Path | None: ...

    @abstractmethod
    def download_dataset_parquet(
        self,
        description: dict | OpenMLDataset,
        download_all_files: bool = False,  # noqa: FBT002
    ) -> Path | None: ...

    @abstractmethod
    def download_dataset_arff(
        self,
        description: dict | OpenMLDataset,
    ) -> Path: ...

    @abstractmethod
    def add_topic(self, dataset_id: int, topic: str) -> int: ...

    @abstractmethod
    def delete_topic(self, dataset_id: int, topic: str) -> int: ...

    @abstractmethod
    def get_online_dataset_format(self, dataset_id: int) -> str: ...

    @abstractmethod
    def get_online_dataset_arff(self, dataset_id: int) -> str | None: ...


class TaskAPI(ResourceAPI):
    """Abstract API interface for task resources."""

    resource_type: ResourceType = ResourceType.TASK


class EvaluationMeasureAPI(ResourceAPI):
    """Abstract API interface for evaluation measure resources."""

    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE

    @abstractmethod
    def list(self) -> list[str]: ...


class EstimationProcedureAPI(ResourceAPI):
    """Abstract API interface for estimation procedure resources."""

    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE

    @abstractmethod
    def list(self) -> list[OpenMLEstimationProcedure]: ...


class EvaluationAPI(ResourceAPI):
    """Abstract API interface for evaluation resources."""

    resource_type: ResourceType = ResourceType.EVALUATION

    @abstractmethod
    def list(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        function: str,
        tasks: list | None = None,
        setups: list | None = None,
        flows: list | None = None,
        runs: list | None = None,
        uploaders: list | None = None,
        study: int | None = None,
        sort_order: str | None = None,
        **kwargs: Any,
    ) -> list[OpenMLEvaluation]: ...


class FlowAPI(ResourceAPI):
    """Abstract API interface for flow resources."""

    resource_type: ResourceType = ResourceType.FLOW

    @abstractmethod
    def get(self, flow_id: int, *, reset_cache: bool = False) -> OpenMLFlow: ...

    @abstractmethod
    def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        uploader: str | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def exists(self, name: str, external_version: str) -> int | bool: ...


class StudyAPI(ResourceAPI):
    """Abstract API interface for study resources."""

    resource_type: ResourceType = ResourceType.STUDY

    @abstractmethod
    def list(  # noqa: PLR0913
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: str | None = None,
        main_entity_type: str | None = None,
        uploader: list[int] | None = None,
        benchmark_suite: int | None = None,
    ) -> pd.DataFrame: ...


class RunAPI(ResourceAPI):
    """Abstract API interface for run resources."""

    resource_type: ResourceType = ResourceType.RUN

    @abstractmethod
    def get(
        self,
        run_id: int,
        *,
        reset_cache: bool = False,
    ) -> OpenMLRun: ...

    @abstractmethod
    def list(  # type: ignore[valid-type]  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        ids: builtins.list[int] | None = None,
        task: builtins.list[int] | None = None,
        setup: builtins.list[int] | None = None,
        flow: builtins.list[int] | None = None,
        uploader: builtins.list[int] | None = None,
        study: int | None = None,
        tag: str | None = None,
        display_errors: bool = False,
        task_type: TaskType | int | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def download_text_file(
        self,
        source: str,
        *,
        md5_checksum: str | None = None,
    ) -> str: ...

    @abstractmethod
    def file_id_to_url(
        self,
        file_id: int,
        filename: str | None = None,
    ) -> str: ...


class SetupAPI(ResourceAPI):
    """Abstract API interface for setup resources."""

    resource_type: ResourceType = ResourceType.SETUP

    @abstractmethod
    def list(
        self,
        limit: int,
        offset: int,
        *,
        setup: Iterable[int] | None = None,
        flow: int | None = None,
        tag: str | None = None,
    ) -> list[OpenMLSetup]: ...

    @abstractmethod
    def get(self, setup_id: int) -> OpenMLSetup: ...

    @abstractmethod
    def exists(
        self,
        flow: OpenMLFlow,
        param_settings: builtins.list[dict[str, Any]],
    ) -> int | bool: ...
