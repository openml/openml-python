from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from requests import Response

import pandas as pd
import xmltodict

import openml.utils
from openml._api.resources.base import DatasetsAPI
from openml.datasets.dataset import OpenMLDataset


class DatasetsV1(DatasetsAPI):
    def get(
        self, dataset_id: int, *, return_response: bool = False
    ) -> OpenMLDataset | tuple[OpenMLDataset, Response]:
        path = f"data/{dataset_id}"
        response = self._http.get(path)
        xml_content = response.text  # .text returns str, .content returns bytes
        dataset = self._create_dataset_from_xml(xml_content)

        if return_response:
            return dataset, response

        return dataset

    def list(  # noqa: PLR0913
        self,
        data_id: list[int] | None = None,
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
    ) -> pd.DataFrame:
        """Return a dataframe of all dataset which are on OpenML.

        Supports large amount of results.

        Parameters
        ----------
        data_id : list, optional
            A list of data ids, to specify which datasets should be
            listed
        offset : int, optional
            The number of datasets to skip, starting from the first.
        size : int, optional
            The maximum number of datasets to show.
        status : str, optional
            Should be {active, in_preparation, deactivated}. By
            default active datasets are returned, but also datasets
            from another status can be requested.
        tag : str, optional
        data_name : str, optional
        data_version : int, optional
        number_instances : int | str, optional
        number_features : int | str, optional
        number_classes : int | str, optional
        number_missing_values : int | str, optional

        Returns
        -------
        datasets: dataframe
            Each row maps to a dataset
            Each column contains the following information:
            - dataset id
            - name
            - format
            - status
            If qualities are calculated for the dataset, some of
            these are also included as columns.
        """
        listing_call = partial(
            self._list_datasets,
            data_id=data_id,
            status=status,
            tag=tag,
            data_name=data_name,
            data_version=data_version,
            number_instances=number_instances,
            number_features=number_features,
            number_classes=number_classes,
            number_missing_values=number_missing_values,
        )
        batches = openml.utils._list_all(listing_call, offset=offset, limit=size)
        if len(batches) == 0:
            return pd.DataFrame()

        return pd.concat(batches)

    def _list_datasets(
        self,
        limit: int,
        offset: int,
        *,
        data_id: list[int] | None = None,  # type: ignore
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Perform api call to return a list of all datasets.

        Parameters
        ----------
        The arguments that are lists are separated from the single value
        ones which are put into the kwargs.
        display_errors is also separated from the kwargs since it has a
        default value.

        limit : int
            The maximum number of datasets to show.
        offset : int
            The number of datasets to skip, starting from the first.
        data_id : list, optional

        kwargs : dict, optional
            Legal filter operators (keys in the dict):
            tag, status, limit, offset, data_name, data_version, number_instances,
            number_features, number_classes, number_missing_values.

        Returns
        -------
        datasets : dataframe
        """
        api_call = "data/list"

        if limit is not None:
            api_call += f"/limit/{limit}"
        if offset is not None:
            api_call += f"/offset/{offset}"

        if kwargs is not None:
            for operator, value in kwargs.items():
                if value is not None:
                    api_call += f"/{operator}/{value}"
        if data_id is not None:
            api_call += f"/data_id/{','.join([str(int(i)) for i in data_id])}"
        return self.__list_datasets(api_call=api_call)

    def __list_datasets(self, api_call: str) -> pd.DataFrame:
        xml_string = self._http.get(api_call).text
        datasets_dict = xmltodict.parse(xml_string, force_list=("oml:dataset",))

        # Minimalistic check if the XML is useful
        assert isinstance(datasets_dict["oml:data"]["oml:dataset"], list), type(
            datasets_dict["oml:data"],
        )
        assert datasets_dict["oml:data"]["@xmlns:oml"] == "http://openml.org/openml", datasets_dict[
            "oml:data"
        ]["@xmlns:oml"]

        datasets = {}
        for dataset_ in datasets_dict["oml:data"]["oml:dataset"]:
            ignore_attribute = ["oml:file_id", "oml:quality"]
            dataset = {
                k.replace("oml:", ""): v for (k, v) in dataset_.items() if k not in ignore_attribute
            }
            dataset["did"] = int(dataset["did"])
            dataset["version"] = int(dataset["version"])

            # The number of qualities can range from 0 to infinity
            for quality in dataset_.get("oml:quality", []):
                try:
                    dataset[quality["@name"]] = int(quality["#text"])
                except ValueError:
                    dataset[quality["@name"]] = float(quality["#text"])
            datasets[dataset["did"]] = dataset

        return pd.DataFrame.from_dict(datasets, orient="index").astype(
            {
                "did": int,
                "version": int,
                "status": pd.CategoricalDtype(["active", "deactivated", "in_preparation"]),
            }
        )

    def _create_dataset_from_xml(self, xml: str) -> OpenMLDataset:
        """Create a dataset given a xml string.

        Parameters
        ----------
        xml : string
            Dataset xml representation.

        Returns
        -------
        OpenMLDataset
        """
        description = xmltodict.parse(xml)["oml:data_set_description"]

        # TODO file path after download, cache_format default = 'pickle'
        arff_file = None
        features_file = None
        parquet_file = None
        qualities_file = None

        return OpenMLDataset(
            description["oml:name"],
            description.get("oml:description"),
            data_format=description["oml:format"],
            dataset_id=int(description["oml:id"]),
            version=int(description["oml:version"]),
            creator=description.get("oml:creator"),
            contributor=description.get("oml:contributor"),
            collection_date=description.get("oml:collection_date"),
            upload_date=description.get("oml:upload_date"),
            language=description.get("oml:language"),
            licence=description.get("oml:licence"),
            url=description["oml:url"],
            default_target_attribute=description.get("oml:default_target_attribute"),
            row_id_attribute=description.get("oml:row_id_attribute"),
            ignore_attribute=description.get("oml:ignore_attribute"),
            version_label=description.get("oml:version_label"),
            citation=description.get("oml:citation"),
            tag=description.get("oml:tag"),
            visibility=description.get("oml:visibility"),
            original_data_url=description.get("oml:original_data_url"),
            paper_url=description.get("oml:paper_url"),
            update_comment=description.get("oml:update_comment"),
            md5_checksum=description.get("oml:md5_checksum"),
            data_file=str(arff_file) if arff_file is not None else None,
            features_file=str(features_file) if features_file is not None else None,
            qualities_file=str(qualities_file) if qualities_file is not None else None,
            parquet_url=description.get("oml:parquet_url"),
            parquet_file=str(parquet_file) if parquet_file is not None else None,
        )


class DatasetsV2(DatasetsAPI):
    def get(
        self, dataset_id: int, *, return_response: bool = False
    ) -> OpenMLDataset | tuple[OpenMLDataset, Response]:
        path = f"datasets/{dataset_id}"
        response = self._http.get(path)
        json_content = response.json()
        dataset = self._create_dataset_from_json(json_content)

        if return_response:
            return dataset, response

        return dataset

    def list(  # noqa: PLR0913
        self,
        data_id: list[int] | None = None,
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
    ) -> pd.DataFrame:
        """Return a dataframe of all dataset which are on OpenML.

        Supports large amount of results.

        Parameters
        ----------
        data_id : list, optional
            A list of data ids, to specify which datasets should be
            listed
        offset : int, optional
            The number of datasets to skip, starting from the first.
        size : int, optional
            The maximum number of datasets to show.
        status : str, optional
            Should be {active, in_preparation, deactivated}. By
            default active datasets are returned, but also datasets
            from another status can be requested.
        tag : str, optional
        data_name : str, optional
        data_version : int, optional
        number_instances : int | str, optional
        number_features : int | str, optional
        number_classes : int | str, optional
        number_missing_values : int | str, optional

        Returns
        -------
        datasets: dataframe
            Each row maps to a dataset
            Each column contains the following information:
            - dataset id
            - name
            - format
            - status
            If qualities are calculated for the dataset, some of
            these are also included as columns.
        """
        listing_call = partial(
            self._list_datasets,
            data_id=data_id,
            status=status,
            tag=tag,
            data_name=data_name,
            data_version=data_version,
            number_instances=number_instances,
            number_features=number_features,
            number_classes=number_classes,
            number_missing_values=number_missing_values,
        )
        batches = openml.utils._list_all(listing_call, offset=offset, limit=size)
        if len(batches) == 0:
            return pd.DataFrame()

        return pd.concat(batches)

    def _list_datasets(
        self,
        limit: int,
        offset: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Perform api call to return a list of all datasets.

        Parameters
        ----------
        The arguments that are lists are separated from the single value
        ones which are put into the kwargs.
        display_errors is also separated from the kwargs since it has a
        default value.

        limit : int
            The maximum number of datasets to show.
        offset : int
            The number of datasets to skip, starting from the first.
        data_id : list, optional

        kwargs : dict, optional
            Legal filter operators (keys in the dict):
            tag, status, limit, offset, data_name, data_version, number_instances,
            number_features, number_classes, number_missing_values, data_id.

        Returns
        -------
        datasets : dataframe
        """
        json: dict[str, Any] = {"pagination": {}}

        if limit is not None:
            json["pagination"]["limit"] = limit
        if offset is not None:
            json["pagination"]["offset"] = offset

        if kwargs is not None:
            for operator, value in kwargs.items():
                if value is not None:
                    json[operator] = value

        return self.__list_datasets(json=json)

    def __list_datasets(self, json: dict) -> pd.DataFrame:
        api_call = "datasets/list"
        datasets_list = self._http.post(api_call, json=json).json()

        # Minimalistic check if the JSON is useful
        assert isinstance(datasets_list, list), type(datasets_list)

        datasets = {}
        for dataset_ in datasets_list:
            ignore_attribute = ["file_id", "quality"]
            dataset = {k: v for (k, v) in dataset_.items() if k not in ignore_attribute}
            dataset["did"] = int(dataset["did"])
            dataset["version"] = int(dataset["version"])

            # The number of qualities can range from 0 to infinity
            for quality in dataset_.get("quality", []):
                try:
                    dataset[quality["name"]] = int(quality["text"])
                except ValueError:
                    dataset[quality["name"]] = float(quality["text"])
            datasets[dataset["did"]] = dataset

        return pd.DataFrame.from_dict(datasets, orient="index").astype(
            {
                "did": int,
                "version": int,
                "status": pd.CategoricalDtype(["active", "deactivated", "in_preparation"]),
            }
        )

    def _create_dataset_from_json(self, json_content: dict) -> OpenMLDataset:
        """Create a dataset given a json.

        Parameters
        ----------
        json_content : dict
            Dataset dict/json representation.

        Returns
        -------
        OpenMLDataset
        """
        # TODO file path after download, cache_format default = 'pickle'
        arff_file = None
        features_file = None
        parquet_file = None
        qualities_file = None

        return OpenMLDataset(
            json_content["name"],
            json_content.get("description"),
            data_format=json_content["format"],
            dataset_id=int(json_content["id"]),
            version=int(json_content["version"]),
            creator=json_content.get("creator"),
            contributor=json_content.get("contributor"),
            collection_date=json_content.get("collection_date"),
            upload_date=json_content.get("upload_date"),
            language=json_content.get("language"),
            licence=json_content.get("licence"),
            url=json_content["url"],
            default_target_attribute=json_content.get("default_target_attribute"),
            row_id_attribute=json_content.get("row_id_attribute"),
            ignore_attribute=json_content.get("ignore_attribute"),
            version_label=json_content.get("version_label"),
            citation=json_content.get("citation"),
            tag=json_content.get("tag"),
            visibility=json_content.get("visibility"),
            original_data_url=json_content.get("original_data_url"),
            paper_url=json_content.get("paper_url"),
            update_comment=json_content.get("update_comment"),
            md5_checksum=json_content.get("md5_checksum"),
            data_file=str(arff_file) if arff_file is not None else None,
            features_file=str(features_file) if features_file is not None else None,
            qualities_file=str(qualities_file) if qualities_file is not None else None,
            parquet_url=json_content.get("parquet_url"),
            parquet_file=str(parquet_file) if parquet_file is not None else None,
        )
