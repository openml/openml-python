from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml._api.http import HTTPClient
    from openml.datasets.dataset import OpenMLDataset
    from openml.tasks.task import OpenMLTask


class ResourceAPI:
    def __init__(self, http: HTTPClient):
        self._http = http


class DatasetsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        dataset_id: int | str,
        version: int | None = None,
        error_if_multiple: bool = False,  # noqa: FBT002, FBT001
        *,
        return_response: bool = False,
    ) -> OpenMLDataset | tuple[OpenMLDataset, Response]: ...

    @abstractmethod
    def list(  # noqa: PLR0913
        self,
        data_id: list[int] | None = None,  # type: ignore
        offset: int | None = None,
        size: int | None = None,
        status: str | None = None,
        tag: str | None = None,
        data_name: str | None = None,
        data_version: int | None = None,
        number_instances: int | str | None = None,
        number_features: int | str | None = None,
        number_classes: int | str | None = None,
        number_missing_values: int | str | None = None,
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
    def fork(self, data_id: int) -> int: ...

    @abstractmethod
    def status_update(self, data_id: int, status: Literal["active", "deactivated"]) -> None: ...

    def _name_to_id(
        self,
        dataset_name: str,
        version: int | None = None,
        error_if_multiple: bool = False,  # noqa: FBT001, FBT002
    ) -> int:
        """Attempt to find the dataset id of the dataset with the given name.

        If multiple datasets with the name exist, and ``error_if_multiple`` is ``False``,
        then return the least recent still active dataset.

        Raises an error if no dataset with the name is found.
        Raises an error if a version is specified but it could not be found.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset for which to find its id.
        version : int, optional
            Version to retrieve. If not specified, the oldest active version is returned.
        error_if_multiple : bool (default=False)
            If `False`, if multiple datasets match, return the least recent active dataset.
            If `True`, if multiple datasets match, raise an error.
        download_qualities : bool, optional (default=True)
            If `True`, also download qualities.xml file. If False it skip the qualities.xml.

        Returns
        -------
        int
        The id of the dataset.
        """
        status = None if version is not None else "active"
        candidates = self.list(
            data_name=dataset_name,
            status=status,
            data_version=version,
        )
        if error_if_multiple and len(candidates) > 1:
            msg = f"Multiple active datasets exist with name '{dataset_name}'."
            raise ValueError(msg)

        if candidates.empty:
            no_dataset_for_name = f"No active datasets exist with name '{dataset_name}'"
            and_version = f" and version '{version}'." if version is not None else "."
            raise RuntimeError(no_dataset_for_name + and_version)

        # Dataset ids are chronological so we can just sort based on ids (instead of version)
        return candidates["did"].min()  # type: ignore


class TasksAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        task_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLTask | tuple[OpenMLTask, Response]: ...
