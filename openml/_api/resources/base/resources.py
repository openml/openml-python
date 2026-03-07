from __future__ import annotations

import builtins
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from openml.datasets.dataset import OpenMLDataFeature, OpenMLDataset
from openml.enums import ResourceType

from .base import ResourceAPI


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
    def get_features(self, dataset_id: int) -> dict[int, OpenMLDataFeature]: ...

    @abstractmethod
    def get_qualities(self, dataset_id: int) -> dict[str, float] | None: ...

    @abstractmethod
    def parse_features_file(
        self, features_file: Path, features_pickle_file: Path
    ) -> dict[int, OpenMLDataFeature]: ...

    @abstractmethod
    def parse_qualities_file(
        self, qualities_file: Path, qualities_pickle_file: Path
    ) -> dict[str, float]: ...

    @abstractmethod
    def _download_file(self, url_ext: str, file_path: str, encoding: str = "utf-8") -> Path: ...

    @abstractmethod
    def download_features_file(self, dataset_id: int) -> Path: ...

    @abstractmethod
    def download_qualities_file(self, dataset_id: int) -> Path: ...

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


class RunAPI(ResourceAPI):
    """Abstract API interface for run resources."""

    resource_type: ResourceType = ResourceType.RUN


class SetupAPI(ResourceAPI):
    """Abstract API interface for setup resources."""

    resource_type: ResourceType = ResourceType.SETUP
