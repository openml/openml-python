from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing_extensions import Literal

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml._api.http import HTTPClient
    from openml.datasets.dataset import OpenMLDataFeature, OpenMLDataset
    from openml.tasks.task import OpenMLTask


class ResourceAPI:
    def __init__(self, http: HTTPClient):
        self._http = http


class DatasetsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        dataset_id: int | str,
        *,
        return_response: bool = False,
    ) -> OpenMLDataset | tuple[OpenMLDataset, Response]: ...

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
    ) -> dict[str, float] | None: ...


class TasksAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        task_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLTask | tuple[OpenMLTask, Response]: ...
