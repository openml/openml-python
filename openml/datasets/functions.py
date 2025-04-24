from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import arff
import pandas as pd

import openml._api_calls

from .dataset import OpenMLDataset

logger = logging.getLogger(__name__)


def create_dataset(
    name: str,
    description: str,
    creator: str,
    contributor: str | None = None,
    collection_date: str | None = None,
    language: str = "English",
    licence: str = "Public",
    default_target_attribute: str | None = None,
    row_id_attribute: str | None = None,
    ignore_attribute: str | list[str] | None = None,
    version_label: str | None = None,
    citation: str | None = None,
    attributes: list | dict | Literal["auto"] = "auto",
    data: pd.DataFrame | None = None,
    parquet_path: str | None = None,
    original_data_url: str | None = None,
    paper_url: str | None = None,
    update_existing: bool = False,
) -> OpenMLDataset:
    if parquet_path is not None:
        if data is not None:
            raise ValueError("Specify either `data` or `parquet_path`, not both.")
        data = pd.read_parquet(parquet_path)
        dataset_format = "parquet"
        file_path = parquet_path
    else:
        if data is None:
            raise ValueError("Either `data` or `parquet_path` must be provided.")
        dataset_format = "arff"
        if attributes == "auto":
            attributes = [(col, "NUMERIC") for col in data.columns]
        arff_object = {
            "relation": name,
            "description": description,
            "attributes": attributes,
            "data": data.values.tolist(),
        }
        file_path = "/tmp/data.arff"
        with open(file_path, "w") as f:
            arff.dump(arff_object, f)

    _upload_to_server(file_path, dataset_format)
    return OpenMLDataset(
        name=name,
        description=description,
        data_format=dataset_format,
        creator=creator,
        contributor=contributor,
        collection_date=collection_date,
        language=language,
        licence=licence,
        default_target_attribute=default_target_attribute,
        row_id_attribute=row_id_attribute,
        ignore_attribute=ignore_attribute,
        citation=citation,
        version_label=version_label,
        original_data_url=original_data_url,
        paper_url=paper_url,
        update_comment=None,
        dataset=file_path,
    )


def _upload_to_server(file_path: str, dataset_format: str) -> str:
    content_type = "application/octet-stream" if dataset_format == "parquet" else "text/plain"

    url = openml._api_calls._create_dataset_upload_url()
    with open(file_path, "rb") as fp:
        response = openml._api_calls._perform_api_call(
            url, "post", file_upload=(Path(file_path).name, fp, content_type)
        )
    return response["id"]
