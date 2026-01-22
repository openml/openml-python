from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml._api.clients.http import HTTPClient
    from openml._api.clients.minio import MinIOClient
    from openml.datasets.dataset import OpenMLDataFeature, OpenMLDataset
    from openml.tasks.task import OpenMLTask


class ResourceAPI:
    def __init__(self, http: HTTPClient):
        self._http = http


class DatasetsAPI(ResourceAPI, ABC):
    def __init__(self, http: HTTPClient, minio: MinIOClient):
        self._minio = minio
        super().__init__(http)

    @abstractmethod
    def get(  # noqa: PLR0913
        self,
        dataset_id: int,
        download_data: bool = False,  # noqa: FBT002
        cache_format: Literal["pickle", "feather"] = "pickle",
        download_qualities: bool = False,  # noqa: FBT002
        download_features_meta_data: bool = False,  # noqa: FBT002
        download_all_files: bool = False,  # noqa: FBT002
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
    def delete(self, dataset_id: int) -> bool: ...

    @abstractmethod
    def edit(  # noqa: PLR0913
        self,
        data_id: int,
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
    def list_qualities(self) -> list[str]: ...  # type: ignore

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
    def add_topic(self, data_id: int, topic: str) -> int: ...

    @abstractmethod
    def delete_topic(self, data_id: int, topic: str) -> int: ...

    @abstractmethod
    def get_online_dataset_format(self, dataset_id: int) -> str: ...

    @abstractmethod
    def get_online_dataset_arff(self, dataset_id: int) -> str | None: ...


class TasksAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        task_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLTask | tuple[OpenMLTask, Response]: ...
