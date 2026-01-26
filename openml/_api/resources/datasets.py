from __future__ import annotations

from typing import TYPE_CHECKING

from openml._api.resources.base import DatasetsAPI, ResourceV1, ResourceV2

if TYPE_CHECKING:
    from responses import Response

    from openml.datasets.dataset import OpenMLDataset


class DatasetsV1(ResourceV1, DatasetsAPI):
    def get(self, dataset_id: int) -> OpenMLDataset | tuple[OpenMLDataset, Response]:
        raise NotImplementedError


class DatasetsV2(ResourceV2, DatasetsAPI):
    def get(self, dataset_id: int) -> OpenMLDataset | tuple[OpenMLDataset, Response]:
        raise NotImplementedError
