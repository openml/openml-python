# ruff: noqa: PLR0913
from __future__ import annotations

import builtins
import json
import logging
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import minio
import pandas as pd
import urllib3
import xmltodict

import openml
from openml.datasets.data_feature import OpenMLDataFeature
from openml.datasets.dataset import OpenMLDataset
from openml.exceptions import (
    OpenMLHashException,
    OpenMLPrivateDatasetError,
    OpenMLServerException,
)

from .base import DatasetAPI, ResourceV1API, ResourceV2API

logger = logging.getLogger(__name__)


NO_ACCESS_GRANTED_ERRCODE = 112


class DatasetV1API(ResourceV1API, DatasetAPI):
    """Version 1 API implementation for dataset resources."""

    @openml.utils.thread_safe_if_oslo_installed
    def get(
        self,
        dataset_id: int,
        download_data: bool = False,  # noqa: FBT002
        cache_format: Literal["pickle", "feather"] = "pickle",
        download_qualities: bool = False,  # noqa: FBT002
        download_features_meta_data: bool = False,  # noqa: FBT002
        download_all_files: bool = False,  # noqa: FBT002
        force_refresh_cache: bool = False,  # noqa: FBT002
    ) -> OpenMLDataset:
        """Download the OpenML dataset representation, optionally also download actual data file.

        Parameters
        ----------
        dataset_id : int or str
            Dataset ID (integer) or dataset name (string) of the dataset to download.
        download_data : bool (default=False)
            If True, download the data file.
        cache_format : str (default='pickle') in {'pickle', 'feather'}
            Format for caching the dataset - may be feather or pickle
            Note that the default 'pickle' option may load slower than feather when
            no.of.rows is very high.
        download_qualities : bool (default=False)
            Option to download 'qualities' meta-data with the minimal dataset description.
        download_features_meta_data : bool (default=False)
            Option to download 'features' meta-data with the minimal dataset description.
        download_all_files: bool (default=False)
            EXPERIMENTAL. Download all files related to the dataset that reside on the server.
        force_refresh_cache : bool (default=False)
            Force the cache to delete the cache directory and re-download the data.

        Returns
        -------
        dataset : :class:`openml.OpenMLDataset`
            The downloaded dataset.
        """
        path = f"data/{dataset_id}"
        try:
            response = self._http.get(path, enable_cache=True, refresh_cache=force_refresh_cache)
            xml_content = response.text
            description = xmltodict.parse(xml_content)["oml:data_set_description"]

            features_file = None
            qualities_file = None

            if download_features_meta_data:
                features_file = self.download_features_file(dataset_id)
            if download_qualities:
                qualities_file = self.download_qualities_file(dataset_id)

            parquet_file = None
            skip_parquet = (
                os.environ.get(openml.config.OPENML_SKIP_PARQUET_ENV_VAR, "false").casefold()
                == "true"
            )
            download_parquet = "oml:parquet_url" in description and not skip_parquet
            if download_parquet and (download_data or download_all_files):
                try:
                    parquet_file = self.download_dataset_parquet(
                        description,
                        download_all_files=download_all_files,
                    )
                except urllib3.exceptions.MaxRetryError:
                    parquet_file = None

            arff_file = None
            if parquet_file is None and download_data:
                if download_parquet:
                    logger.warning("Failed to download parquet, fallback on ARFF.")
                arff_file = self.download_dataset_arff(description)
        except OpenMLServerException as e:
            # if there was an exception
            # check if the user had access to the dataset
            if e.code == NO_ACCESS_GRANTED_ERRCODE:
                raise OpenMLPrivateDatasetError(e.message) from None

            raise e

        return self._create_dataset_from_xml(
            description, features_file, qualities_file, arff_file, parquet_file, cache_format
        )

    def list(
        self,
        limit: int,
        offset: int,
        *,
        data_id: builtins.list[int] | None = None,
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
        xml_string = self._http.get(api_call).text
        return self._parse_list_xml(xml_string)

    def edit(
        self,
        dataset_id: int,
        description: str | None = None,
        creator: str | None = None,
        contributor: str | None = None,
        collection_date: str | None = None,
        language: str | None = None,
        default_target_attribute: str | None = None,
        ignore_attribute: str | builtins.list[str] | None = None,
        citation: str | None = None,
        row_id_attribute: str | None = None,
        original_data_url: str | None = None,
        paper_url: str | None = None,
    ) -> int:
        """Edits an OpenMLDataset.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.
        description : str, optional
            Description of the dataset.
        creator : str, optional
            The person who created the dataset.
        contributor : str, optional
            People who contributed to the current version of the dataset.
        collection_date : str, optional
            The date the data was originally collected, given by the uploader.
        language : str, optional
            Language in which the data is represented.
            Starts with 1 upper case letter, rest lower case, e.g. 'English'.
        default_target_attribute : str, optional
            The default target attribute, if it exists.
            Can have multiple values, comma separated.
        ignore_attribute : str | list, optional
            Attributes that should be excluded in modelling,
            such as identifiers and indexes.
        citation : str, optional
            Reference(s) that should be cited when building on this data.
        row_id_attribute : str, optional
            The attribute that represents the row-id column, if present in the
            dataset. If ``data`` is a dataframe and ``row_id_attribute`` is not
            specified, the index of the dataframe will be used as the
            ``row_id_attribute``. If the name of the index is ``None``, it will
            be discarded.

            .. versionadded: 0.8
                Inference of ``row_id_attribute`` from a dataframe.
        original_data_url : str, optional
            For derived data, the url to the original dataset.
        paper_url : str, optional
            Link to a paper describing the dataset.

        Returns
        -------
        Dataset id
        """
        # compose data edit parameters as xml
        form_data = {"data_id": dataset_id}  # type: dict[str, str | int]
        xml = OrderedDict()  # type: 'OrderedDict[str, OrderedDict]'
        xml["oml:data_edit_parameters"] = OrderedDict()
        xml["oml:data_edit_parameters"]["@xmlns:oml"] = "http://openml.org/openml"
        xml["oml:data_edit_parameters"]["oml:description"] = description
        xml["oml:data_edit_parameters"]["oml:creator"] = creator
        xml["oml:data_edit_parameters"]["oml:contributor"] = contributor
        xml["oml:data_edit_parameters"]["oml:collection_date"] = collection_date
        xml["oml:data_edit_parameters"]["oml:language"] = language
        xml["oml:data_edit_parameters"]["oml:default_target_attribute"] = default_target_attribute
        xml["oml:data_edit_parameters"]["oml:row_id_attribute"] = row_id_attribute
        xml["oml:data_edit_parameters"]["oml:ignore_attribute"] = ignore_attribute
        xml["oml:data_edit_parameters"]["oml:citation"] = citation
        xml["oml:data_edit_parameters"]["oml:original_data_url"] = original_data_url
        xml["oml:data_edit_parameters"]["oml:paper_url"] = paper_url

        # delete None inputs
        for k in list(xml["oml:data_edit_parameters"]):
            if not xml["oml:data_edit_parameters"][k]:
                del xml["oml:data_edit_parameters"][k]

        file_elements = {
            "edit_parameters": ("description.xml", xmltodict.unparse(xml)),
        }  # type: dict[str, str | tuple[str, str]]
        result_xml = self._http.post("data/edit", data=form_data, files=file_elements).text
        result = xmltodict.parse(result_xml)
        dataset_id = result["oml:data_edit"]["oml:id"]
        return int(dataset_id)

    def fork(self, dataset_id: int) -> int:
        """
        Creates a new dataset version, with the authenticated user as the new owner.
        The forked dataset can have distinct dataset meta-data,
        but the actual data itself is shared with the original version.

        Parameters
        ----------
        dataset_id : int
            id of the dataset to be forked

        Returns
        -------
        Dataset id of the forked dataset

        """
        # compose data fork parameters
        form_data = {"data_id": dataset_id}
        result_xml = self._http.post("data/fork", data=form_data).text
        result = xmltodict.parse(result_xml)
        dataset_id = result["oml:data_fork"]["oml:id"]
        return int(dataset_id)

    def status_update(self, dataset_id: int, status: Literal["active", "deactivated"]) -> None:
        """
        Updates the status of a dataset to either 'active' or 'deactivated'.
        Please see the OpenML API documentation for a description of the status
        and all legal status transitions:
        https://docs.openml.org/concepts/data/#dataset-status

        Parameters
        ----------
        dataset_id : int
            The data id of the dataset
        status : str,
            'active' or 'deactivated'
        """
        legal_status = {"active", "deactivated"}
        if status not in legal_status:
            raise ValueError(f"Illegal status value. Legal values: {legal_status}")

        data: dict[str, str | int] = {"data_id": dataset_id, "status": status}
        result_xml = self._http.post("data/status/update", data=data).text
        result = xmltodict.parse(result_xml)
        server_data_id = result["oml:data_status_update"]["oml:id"]
        server_status = result["oml:data_status_update"]["oml:status"]
        if status != server_status or int(dataset_id) != int(server_data_id):
            # This should never happen
            raise ValueError("Data id/status does not collide")

    def list_qualities(self) -> builtins.list[str]:
        """Return list of data qualities available.

        The function performs an API call to retrieve the entire list of
        data qualities that are computed on the datasets uploaded.

        Returns
        -------
        list
        """
        api_call = "data/qualities/list"
        xml_string = self._http.get(api_call).text
        qualities = xmltodict.parse(xml_string, force_list=("oml:quality"))
        # Minimalistic check if the XML is useful
        if "oml:data_qualities_list" not in qualities:
            raise ValueError('Error in return XML, does not contain "oml:data_qualities_list"')

        if not isinstance(qualities["oml:data_qualities_list"]["oml:quality"], list):
            raise TypeError('Error in return XML, does not contain "oml:quality" as a list')

        return qualities["oml:data_qualities_list"]["oml:quality"]

    def _create_dataset_from_xml(
        self,
        description: dict,
        features_file: Path | None = None,
        qualities_file: Path | None = None,
        arff_file: Path | None = None,
        parquet_file: Path | None = None,
        cache_format: Literal["pickle", "feather"] = "pickle",
    ) -> OpenMLDataset:
        """Create a dataset given a parsed xml dict.

        Parameters
        ----------
        description : dict
            Parsed xml dict representing the dataset description.
        features_file : Path, optional
            Path to features file.
        qualities_file : Path, optional
            Path to qualities file.
        arff_file : Path, optional
            Path to arff file.
        parquet_file : Path, optional
            Path to parquet file.
        cache_format : str (default='pickle') in {'pickle', 'feather'}
            Format for caching the dataset - may be feather or pickle

        Returns
        -------
        OpenMLDataset
        """
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
            cache_format=cache_format,
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

    def feature_add_ontology(self, dataset_id: int, index: int, ontology: str) -> bool:
        """
        An ontology describes the concept that are described in a feature. An
        ontology is defined by an URL where the information is provided. Adds
        an ontology (URL) to a given dataset feature (defined by a dataset id
        and index). The dataset has to exists on OpenML and needs to have been
        processed by the evaluation engine.

        Parameters
        ----------
        dataset_id : int
            id of the dataset to which the feature belongs
        index : int
            index of the feature in dataset (0-based)
        ontology : str
            URL to ontology (max. 256 characters)

        Returns
        -------
        True or throws an OpenML server exception
        """
        upload_data: dict[str, int | str] = {
            "data_id": dataset_id,
            "index": index,
            "ontology": ontology,
        }
        self._http.post("data/feature/ontology/add", data=upload_data)
        # an error will be thrown in case the request was unsuccessful
        return True

    def feature_remove_ontology(self, dataset_id: int, index: int, ontology: str) -> bool:
        """
        Removes an existing ontology (URL) from a given dataset feature (defined
        by a dataset id and index). The dataset has to exists on OpenML and needs
        to have been processed by the evaluation engine. Ontology needs to be
        attached to the specific fearure.

        Parameters
        ----------
        dataset_id : int
            id of the dataset to which the feature belongs
        index : int
            index of the feature in dataset (0-based)
        ontology : str
            URL to ontology (max. 256 characters)

        Returns
        -------
        True or throws an OpenML server exception
        """
        upload_data: dict[str, int | str] = {
            "data_id": dataset_id,
            "index": index,
            "ontology": ontology,
        }
        self._http.post("data/feature/ontology/remove", data=upload_data)
        # an error will be thrown in case the request was unsuccessful
        return True

    def get_features(self, dataset_id: int) -> dict[int, OpenMLDataFeature]:
        """Get features of a dataset from server.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        dict[int, OpenMLDataFeature]
        """
        path = f"data/features/{dataset_id}"
        xml = self._http.get(path, enable_cache=True).text
        _ = self.download_features_file(dataset_id)  # ensure the file is downloaded and cached
        return self._parse_features_xml(xml)

    def get_qualities(self, dataset_id: int) -> dict[str, float] | None:
        """Get qualities of a dataset from server.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        dict[str, float] | None
        """
        path = f"data/qualities/{dataset_id!s}"
        try:
            xml = self._http.get(path, enable_cache=True).text
        except OpenMLServerException as e:
            if e.code == 362 and str(e) == "No qualities found - None":
                # quality file stays as None
                logger.warning(f"No qualities found for dataset {dataset_id}")
                return None

            raise e
        _ = self.download_qualities_file(dataset_id)  # ensure the file is downloaded and cached
        return self._parse_qualities_xml(xml)

    def parse_features_file(
        self, features_file: Path, features_pickle_file: Path | None = None
    ) -> dict[int, OpenMLDataFeature]:
        """
        Parse features file (xml) and store it as a pickle file.

        Parameters
        ----------
        features_file : Path
            Path to features file.
        features_pickle_file : Path, optional
            Path to pickle file for features.

        Returns
        -------
        features : dict[int, OpenMLDataFeature]
        """
        if features_pickle_file is None:
            features_pickle_file = features_file.with_suffix(features_file.suffix + ".pkl")
        assert features_file.suffix == ".xml"

        with Path(features_file).open("r", encoding="utf8") as fh:
            features_xml = fh.read()

        features = self._parse_features_xml(features_xml)

        with features_pickle_file.open("wb") as fh_binary:
            pickle.dump(features, fh_binary)

        return features

    def parse_qualities_file(
        self, qualities_file: Path, qualities_pickle_file: Path | None = None
    ) -> dict[str, float]:
        """Parse qualities file (xml) and store it as a pickle file.

        Parameters
        ----------
        qualities_file : Path
            Path to qualities file.
        qualities_pickle_file : Path, optional
            Path to pickle file for qualities.

        Returns
        -------
        qualities : dict[str, float]
        """
        if qualities_pickle_file is None:
            qualities_pickle_file = qualities_file.with_suffix(qualities_file.suffix + ".pkl")
        assert qualities_file.suffix == ".xml"

        with Path(qualities_file).open("r", encoding="utf8") as fh:
            qualities_xml = fh.read()

        qualities = self._parse_qualities_xml(qualities_xml)

        with qualities_pickle_file.open("wb") as fh_binary:
            pickle.dump(qualities, fh_binary)

        return qualities

    def _parse_features_xml(self, features_xml_string: str) -> dict[int, OpenMLDataFeature]:
        """Parse features xml string.

        Parameters
        ----------
        features_xml_string : str
            Features xml string.

        Returns
        -------
        features : dict[int, OpenMLDataFeature]
        """
        xml_dict = xmltodict.parse(
            features_xml_string,
            force_list=("oml:feature", "oml:nominal_value"),
            strip_whitespace=False,
        )
        features_xml = xml_dict["oml:data_features"]

        features: dict[int, OpenMLDataFeature] = {}
        for idx, xmlfeature in enumerate(features_xml["oml:feature"]):
            nr_missing = xmlfeature.get("oml:number_of_missing_values", 0)
            feature = OpenMLDataFeature(
                int(xmlfeature["oml:index"]),
                xmlfeature["oml:name"],
                xmlfeature["oml:data_type"],
                xmlfeature.get("oml:nominal_value"),
                int(nr_missing),
                xmlfeature.get("oml:ontology"),
            )
            if idx != feature.index:
                raise ValueError("Data features not provided in right order")
            features[feature.index] = feature

        return features

    def _parse_qualities_xml(self, qualities_xml: str) -> dict[str, float]:
        """Parse qualities xml string.

        Parameters
        ----------
        qualities_xml : str
            Qualities xml string.

        Returns
        -------
        qualities : dict[str, float]
        """
        xml_as_dict = xmltodict.parse(qualities_xml, force_list=("oml:quality",))
        qualities = xml_as_dict["oml:data_qualities"]["oml:quality"]
        qualities_ = {}
        for xmlquality in qualities:
            name = xmlquality["oml:name"]
            if xmlquality.get("oml:value", None) is None or xmlquality["oml:value"] == "null":
                value = float("NaN")
            else:
                value = float(xmlquality["oml:value"])
            qualities_[name] = value
        return qualities_

    def _parse_list_xml(self, xml_string: str) -> pd.DataFrame:
        """Parse list response xml string.

        Parameters
        ----------
        xml_string : str
            List response xml string.

        Returns
        -------
        pd.DataFrame
        """
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

    def _download_file(self, url_ext: str) -> Path:
        """Helper method to pass respective handler to downloader.

        Parameters
        ----------
        url_ext : str
            URL extension to download from.

        Returns
        -------
        Path
        """
        self._http.get(url_ext, enable_cache=True)
        return self._http.cache_path_from_url(url_ext)

    def download_features_file(self, dataset_id: int) -> Path:
        """Download features file.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        Path
        """
        path = f"data/features/{dataset_id}"
        file = self._download_file(path)
        self.parse_features_file(file)
        return file

    def download_qualities_file(self, dataset_id: int) -> Path:
        """Download qualities file.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        Path
        """
        path = f"data/qualities/{dataset_id}"
        file = self._download_file(path)
        self.parse_qualities_file(file)
        return file

    def download_dataset_parquet(
        self,
        description: dict | OpenMLDataset,
        download_all_files: bool = False,  # noqa: FBT002
    ) -> Path | None:
        """Download dataset parquet file.

        Parameters
        ----------
        description : dictionary or OpenMLDataset
            Either a dataset description as dict or OpenMLDataset.
        download_all_files: bool, optional (default=False)
            If `True`, download all data found in the bucket to which the description's
            ``parquet_url`` points, only download the parquet file otherwise.

        Returns
        -------
        Path | None
        """
        if isinstance(description, dict):
            url = str(description.get("oml:parquet_url"))
        elif isinstance(description, OpenMLDataset):
            url = str(description._parquet_url)
            assert description.dataset_id is not None
        else:
            raise TypeError("`description` should be either OpenMLDataset or Dict.")

        if download_all_files:
            self._minio.download_minio_bucket(source=url)

        try:
            output_file_path = self._minio.download_minio_file(
                source=url,
            )
        except (FileNotFoundError, urllib3.exceptions.MaxRetryError, minio.error.ServerError) as e:
            logger.warning(f"Could not download file from {url}: {e}")
            return None
        return output_file_path

    def download_dataset_arff(
        self,
        description: dict | OpenMLDataset,
    ) -> Path:
        """Download dataset arff file.

        Parameters
        ----------
        description : dictionary or OpenMLDataset
            Either a dataset description as dict or OpenMLDataset.

        Returns
        -------
        output_filename : Path
            Location of ARFF file.
        """
        if isinstance(description, dict):
            md5_checksum_fixture = description.get("oml:md5_checksum")
            url = str(description["oml:url"])
            did = int(description.get("oml:id"))  # type: ignore
        elif isinstance(description, OpenMLDataset):
            md5_checksum_fixture = description.md5_checksum
            assert description.url is not None
            assert description.dataset_id is not None

            url = description.url
            did = int(description.dataset_id)
        else:
            raise TypeError("`description` should be either OpenMLDataset or Dict.")

        try:
            # save the file in cache and get it's path
            self._http.get(url, enable_cache=True, md5_checksum=md5_checksum_fixture)
            output_file_path = self._http.cache_path_from_url(url)
        except OpenMLHashException as e:
            additional_info = f" Raised when downloading dataset {did}."
            e.args = (e.args[0] + additional_info,)
            raise e

        return output_file_path

    def add_topic(self, dataset_id: int, topic: str) -> int:
        """
        Adds a topic to a dataset.
        This API is not available for all OpenML users and is accessible only by admins.

        Parameters
        ----------
        dataset_id : int
            id of the dataset to be forked
        topic : str
            Topic to be added

        Returns
        -------
        Dataset id
        """
        form_data = {"data_id": dataset_id, "topic": topic}  # type: dict[str, str | int]
        result_xml = self._http.post("data/topicadd", data=form_data).text
        result = xmltodict.parse(result_xml)
        dataset_id = result["oml:data_topic"]["oml:id"]
        return int(dataset_id)

    def delete_topic(self, dataset_id: int, topic: str) -> int:
        """
        Removes a topic from a dataset.
        This API is not available for all OpenML users and is accessible only by admins.

        Parameters
        ----------
        dataset_id : int
            id of the dataset to be forked
        topic : str
            Topic to be deleted

        Returns
        -------
        Dataset id
        """
        form_data = {"data_id": dataset_id, "topic": topic}  # type: dict[str, str | int]
        result_xml = self._http.post("data/topicdelete", data=form_data).text
        result = xmltodict.parse(result_xml)
        dataset_id = result["oml:data_topic"]["oml:id"]
        return int(dataset_id)

    def get_online_dataset_format(self, dataset_id: int) -> str:
        """Get the dataset format for a given dataset id from the OpenML website.

        Parameters
        ----------
        dataset_id : int
            A dataset id.

        Returns
        -------
        str
            Dataset format.
        """
        dataset_xml = self._http.get(f"data/{dataset_id}").text
        # build a dict from the xml and get the format from the dataset description
        return xmltodict.parse(dataset_xml)["oml:data_set_description"]["oml:format"].lower()  # type: ignore

    def get_online_dataset_arff(self, dataset_id: int) -> str | None:
        """Download the ARFF file for a given dataset id
        from the OpenML website.

        Parameters
        ----------
        dataset_id : int
            A dataset id.

        Returns
        -------
        str or None
            A string representation of an ARFF file. Or None if file already exists.
        """
        dataset_xml = self._http.get(f"data/{dataset_id}").text
        # build a dict from the xml.
        # use the url from the dataset description and return the ARFF string
        arff_file = self.download_dataset_arff(
            xmltodict.parse(dataset_xml)["oml:data_set_description"]
        )
        with arff_file.open("r", encoding="utf8") as f:
            return f.read()


class DatasetV2API(ResourceV2API, DatasetAPI):
    """Version 2 API implementation for dataset resources."""

    @openml.utils.thread_safe_if_oslo_installed
    def get(
        self,
        dataset_id: int,
        download_data: bool = False,  # noqa: FBT002
        cache_format: Literal["pickle", "feather"] = "pickle",
        download_qualities: bool = False,  # noqa: FBT002
        download_features_meta_data: bool = False,  # noqa: FBT002
        download_all_files: bool = False,  # noqa: FBT002
        force_refresh_cache: bool = False,  # noqa: FBT002
    ) -> OpenMLDataset:
        """Download the OpenML dataset representation, optionally also download actual data file.

        Parameters
        ----------
        dataset_id : int or str
            Dataset ID (integer) or dataset name (string) of the dataset to download.
        download_data : bool (default=False)
            If True, download the data file.
        cache_format : str (default='pickle') in {'pickle', 'feather'}
            Format for caching the dataset - may be feather or pickle
            Note that the default 'pickle' option may load slower than feather when
            no.of.rows is very high.
        download_qualities : bool (default=False)
            Option to download 'qualities' meta-data with the minimal dataset description.
        download_features_meta_data : bool (default=False)
            Option to download 'features' meta-data with the minimal dataset description.
        download_all_files: bool (default=False)
            EXPERIMENTAL. Download all files related to the dataset that reside on the server.
        force_refresh_cache : bool (default=False)
            Force the cache to delete the cache directory and re-download the data.

        Returns
        -------
        dataset : :class:`openml.OpenMLDataset`
            The downloaded dataset.
        """
        path = f"datasets/{dataset_id}"
        try:
            response = self._http.get(path, enable_cache=True, refresh_cache=force_refresh_cache)
            json_content = response.json()
            features_file = None
            qualities_file = None

            if download_features_meta_data:
                features_file = self.download_features_file(dataset_id)
            if download_qualities:
                qualities_file = self.download_qualities_file(dataset_id)

            parquet_file = None
            skip_parquet = (
                os.environ.get(openml.config.OPENML_SKIP_PARQUET_ENV_VAR, "false").casefold()
                == "true"
            )
            download_parquet = "parquet_url" in json_content and not skip_parquet
            if download_parquet and (download_data or download_all_files):
                try:
                    parquet_file = self.download_dataset_parquet(
                        json_content,
                        download_all_files=download_all_files,
                    )
                except urllib3.exceptions.MaxRetryError:
                    parquet_file = None

            arff_file = None
            if parquet_file is None and download_data:
                if download_parquet:
                    logger.warning("Failed to download parquet, fallback on ARFF.")
                arff_file = self.download_dataset_arff(json_content)
        except OpenMLServerException as e:
            # if there was an exception
            # check if the user had access to the dataset
            if e.code == NO_ACCESS_GRANTED_ERRCODE:
                raise OpenMLPrivateDatasetError(e.message) from None

            raise e

        return self._create_dataset_from_json(
            json_content, features_file, qualities_file, arff_file, parquet_file, cache_format
        )

    def list(
        self,
        limit: int,
        offset: int,
        *,
        data_id: builtins.list[int] | None = None,
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
        data_id: list[int], optional

        kwargs : dict, optional
            Legal filter operators (keys in the dict):
            tag, status, limit, offset, data_name, data_version, number_instances,
            number_features, number_classes, number_missing_values.

        Returns
        -------
        datasets : dataframe
        """
        json: dict[str, Any] = {"pagination": {}}

        if limit is not None:
            json["pagination"]["limit"] = limit
        if offset is not None:
            json["pagination"]["offset"] = offset
        if data_id is not None:
            json["data_id"] = data_id
        if kwargs is not None:
            for operator, value in kwargs.items():
                if value is not None:
                    json[operator] = value

        api_call = "datasets/list"
        datasets_list = self._http.post(path=api_call, json=json, use_api_key=False).json()
        # Minimalistic check if the JSON is useful
        assert isinstance(datasets_list, list), type(datasets_list)

        return self._parse_list_json(datasets_list)

    def edit(
        self,
        dataset_id: int,
        description: str | None = None,
        creator: str | None = None,
        contributor: str | None = None,
        collection_date: str | None = None,
        language: str | None = None,
        default_target_attribute: str | None = None,
        ignore_attribute: str | builtins.list[str] | None = None,
        citation: str | None = None,
        row_id_attribute: str | None = None,
        original_data_url: str | None = None,
        paper_url: str | None = None,
    ) -> int:
        _ = (
            dataset_id,
            description,
            creator,
            contributor,
            collection_date,
            language,
            default_target_attribute,
            ignore_attribute,
            citation,
            row_id_attribute,
            original_data_url,
            paper_url,
        )  # unused method arg mypy error
        raise self._not_supported(method="edit")

    def fork(self, dataset_id: int) -> int:
        _ = dataset_id  # unused method arg mypy error
        raise self._not_supported(method="fork")

    def status_update(self, dataset_id: int, status: Literal["active", "deactivated"]) -> None:
        """
        Updates the status of a dataset to either 'active' or 'deactivated'.
        Please see the OpenML API documentation for a description of the status
        and all legal status transitions:
        https://docs.openml.org/concepts/data/#dataset-status

        Parameters
        ----------
        dataset_id : int
            The data id of the dataset
        status : str,
            'active' or 'deactivated'
        """
        legal_status = {"active", "deactivated"}
        if status not in legal_status:
            raise ValueError(f"Illegal status value. Legal values: {legal_status}")

        data: dict[str, str | int] = {"dataset_id": dataset_id, "status": status}
        # TODO needs fix for api and json
        result = self._http.post(
            f"datasets/status/update/?api_key={self._http.api_key}", json=data, use_api_key=False
        ).json()
        server_data_id = result["dataset_id"]
        server_status = result["status"]
        if status != server_status or int(dataset_id) != int(server_data_id):
            # This should never happen
            raise ValueError("Data id/status does not collide")

    def list_qualities(self) -> builtins.list[str]:
        """Return list of data qualities available.

        The function performs an API call to retrieve the entire list of
        data qualities that are computed on the datasets uploaded.

        Returns
        -------
        list
        """
        api_call = "datasets/qualities/list"
        qualities = self._http.get(api_call).json()
        # Minimalistic check if the XML is useful
        if "data_qualities_list" not in qualities:
            raise ValueError('Error in return XML, does not contain "oml:data_qualities_list"')

        if not isinstance(qualities["data_qualities_list"]["quality"], list):
            raise TypeError('Error in return json, does not contain "quality" as a list')

        return qualities["data_qualities_list"]["quality"]

    def _create_dataset_from_json(
        self,
        json_content: dict,
        features_file: Path | None = None,
        qualities_file: Path | None = None,
        arff_file: Path | None = None,
        parquet_file: Path | None = None,
        cache_format: Literal["pickle", "feather"] = "pickle",
    ) -> OpenMLDataset:
        """Create a dataset given a json.

        Parameters
        ----------
        json_content : dict
            Dataset dict/json representation.
        features_file : Path, optional
            Path to features file.
        qualities_file : Path, optional
            Path to qualities file.
        arff_file : Path, optional
            Path to arff file.
        parquet_file : Path, optional
            Path to parquet file.
        cache_format : str (default='pickle') in {'pickle', 'feather'}
            Format for caching the dataset - may be feather or pickle

        Returns
        -------
        OpenMLDataset
        """
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
            cache_format=cache_format,
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

    def feature_add_ontology(self, dataset_id: int, index: int, ontology: str) -> bool:
        _ = (dataset_id, index, ontology)  # unused method arg mypy error
        raise self._not_supported(method="feature_add_ontology")

    def feature_remove_ontology(self, dataset_id: int, index: int, ontology: str) -> bool:
        _ = (dataset_id, index, ontology)  # unused method arg mypy error
        raise self._not_supported(method="feature_remove_ontology")

    def get_features(self, dataset_id: int) -> dict[int, OpenMLDataFeature]:
        """Get features of a dataset from server.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        dict[int, OpenMLDataFeature]
        Dictionary mapping feature index to OpenMLDataFeature.
        """
        path = f"datasets/features/{dataset_id}"
        json = self._http.get(path, enable_cache=True).json()

        return self._parse_features_json(json)

    def get_qualities(self, dataset_id: int) -> dict[str, float] | None:
        """Get qualities of a dataset from server.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        dict[str, float] | None
        Dictionary mapping quality name to quality value.
        """
        path = f"datasets/qualities/{dataset_id!s}"
        try:
            qualities_json = self._http.get(path, enable_cache=True).json()
        except OpenMLServerException as e:
            if e.code == 362 and str(e) == "No qualities found - None":
                logger.warning(f"No qualities found for dataset {dataset_id}")
                return None

            raise e

        return self._parse_qualities_json(qualities_json)

    def parse_features_file(
        self, features_file: Path, features_pickle_file: Path | None = None
    ) -> dict[int, OpenMLDataFeature]:
        """
        Parse features file (json) and store it as a pickle file.

        Parameters
        ----------
        features_file : Path
            Path to features file.
        features_pickle_file : Path, optional
            Path to pickle file for features.

        Returns
        -------
        dict[int, OpenMLDataFeature]
        """
        if features_pickle_file is None:
            features_pickle_file = features_file.with_suffix(features_file.suffix + ".pkl")
        if features_file.suffix == ".xml":
            # can fallback to v1 if the file is .xml
            raise NotImplementedError("Unable to Parse .xml from v1")

        with Path(features_file).open("r", encoding="utf8") as fh:
            features_json = json.load(fh)

        features = self._parse_features_json(features_json)

        with features_pickle_file.open("wb") as fh_binary:
            pickle.dump(features, fh_binary)

        return features

    def parse_qualities_file(
        self, qualities_file: Path, qualities_pickle_file: Path | None = None
    ) -> dict[str, float]:
        """Parse qualities file (json) and store it as a pickle file.

        Parameters
        ----------
        qualities_file : Path
            Path to qualities file.
        qualities_pickle_file : Path, optional
            Path to pickle file for qualities.

        Returns
        -------
        qualities : dict[str, float]
        """
        if qualities_pickle_file is None:
            qualities_pickle_file = qualities_file.with_suffix(qualities_file.suffix + ".pkl")
        if qualities_file.suffix == ".xml":
            # can fallback to v1 if the file is .xml
            raise NotImplementedError("Unable to Parse .xml from v1")

        with Path(qualities_file).open("r", encoding="utf8") as fh:
            qualities_json = json.load(fh)

        qualities = self._parse_qualities_json(qualities_json)

        with qualities_pickle_file.open("wb") as fh_binary:
            pickle.dump(qualities, fh_binary)

        return qualities

    def _parse_features_json(self, features_json: dict) -> dict[int, OpenMLDataFeature]:
        """Parse features json.

        Parameters
        ----------
        features_json : dict
            Features json.

        Returns
        -------
        dict[int, OpenMLDataFeature]
        """
        features: dict[int, OpenMLDataFeature] = {}
        for idx, jsonfeatures in enumerate(features_json):
            nr_missing = jsonfeatures.get("number_of_missing_values", 0)
            feature = OpenMLDataFeature(
                int(jsonfeatures["index"]),
                jsonfeatures["name"],
                jsonfeatures["data_type"],
                jsonfeatures.get("nominal_values"),
                int(nr_missing),
                jsonfeatures.get("ontology"),
            )
            if idx != feature.index:
                raise ValueError("Data features not provided in right order")
            features[feature.index] = feature

        return features

    def _parse_qualities_json(self, qualities_json: dict) -> dict[str, float]:
        """Parse qualities json.

        Parameters
        ----------
        qualities_json : dict
            Qualities json.

        Returns
        -------
        dict[str, float]
        """
        qualities_ = {}
        for quality in qualities_json:
            name = quality["name"]
            if quality.get("value", None) is None or quality["value"] == "null":
                value = float("NaN")
            else:
                value = float(quality["value"])
            qualities_[name] = value
        return qualities_

    def _parse_list_json(self, datasets_list: builtins.list) -> pd.DataFrame:
        """Parse list response json.

        Parameters
        ----------
        datasets_list : list
            List of datasets in json format.

        Returns
        -------
        pd.DataFrame
        """
        datasets = {}
        for dataset_ in datasets_list:
            ignore_attribute = ["file_id", "quality", "md5_checksum"]
            dataset = {k: v for (k, v) in dataset_.items() if k not in ignore_attribute}
            dataset["did"] = int(dataset["did"])
            dataset["version"] = int(dataset["version"])

            # The number of qualities can range from 0 to infinity
            for quality in dataset_.get("quality", []):
                try:
                    dataset[quality["name"]] = int(quality["value"])
                except ValueError:
                    dataset[quality["name"]] = float(quality["value"])
            datasets[dataset["did"]] = dataset

        return pd.DataFrame.from_dict(datasets, orient="index").astype(
            {
                "did": int,
                "version": int,
                "status": pd.CategoricalDtype(["active", "deactivated", "in_preparation"]),
            }
        )

    def _download_file(self, url_ext: str) -> Path:
        """Helper method to pass respective handler to downloader.

        Parameters
        ----------
        url_ext : str
            URL extension to download from.

        Returns
        -------
        Path
        """
        self._http.get(url_ext, enable_cache=True)
        return self._http.cache_path_from_url(url_ext)

    def download_features_file(self, dataset_id: int) -> Path:
        """Download features file.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        Path
        """
        path = f"datasets/features/{dataset_id}"
        file = self._download_file(path)
        self.parse_features_file(file)
        return file

    def download_qualities_file(self, dataset_id: int) -> Path:
        """Download qualities file.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.

        Returns
        -------
        Path
        """
        path = f"datasets/qualities/{dataset_id}"
        file = self._download_file(path)
        self.parse_qualities_file(file)
        return file

    def download_dataset_parquet(
        self,
        description: dict | OpenMLDataset,
        download_all_files: bool = False,  # noqa: FBT002
    ) -> Path | None:
        """Download dataset parquet file.

        Parameters
        ----------
        description : dictionary or OpenMLDataset
            Either a dataset description as dict or OpenMLDataset.
        download_all_files: bool, optional (default=False)
            If `True`, download all data found in the bucket to which the description's
            ``parquet_url`` points, only download the parquet file otherwise.

        Returns
        -------
        Path | None
        """
        if isinstance(description, dict):
            url = str(description.get("parquet_url"))
        elif isinstance(description, OpenMLDataset):
            url = str(description._parquet_url)
            assert description.dataset_id is not None
        else:
            raise TypeError("`description` should be either OpenMLDataset or Dict.")

        if download_all_files:
            self._minio.download_minio_bucket(source=url)

        try:
            output_file_path = self._minio.download_minio_file(source=url)
        except (FileNotFoundError, urllib3.exceptions.MaxRetryError, minio.error.ServerError) as e:
            logger.warning(f"Could not download file from {url}: {e}")
            return None
        return output_file_path

    def download_dataset_arff(
        self,
        description: dict | OpenMLDataset,
    ) -> Path:
        """Download dataset arff file.

        Parameters
        ----------
        description : dictionary or OpenMLDataset
            Either a dataset description as dict or OpenMLDataset.

        Returns
        -------
        output_filename : Path
            Location of ARFF file.
        """
        if isinstance(description, dict):
            url = str(description["url"])
            did = int(description.get("id"))  # type: ignore
        elif isinstance(description, OpenMLDataset):
            assert description.url is not None
            assert description.dataset_id is not None

            url = description.url
            did = int(description.dataset_id)
        else:
            raise TypeError("`description` should be either OpenMLDataset or Dict.")

        try:
            # save the file in cache and get it's path
            self._http.get(url, enable_cache=True)
            output_file_path = self._http.cache_path_from_url(url)
        except OpenMLHashException as e:
            additional_info = f" Raised when downloading dataset {did}."
            e.args = (e.args[0] + additional_info,)
            raise e

        return output_file_path

    def add_topic(self, dataset_id: int, topic: str) -> int:
        _ = (dataset_id, topic)  # unused method arg mypy error
        raise self._not_supported(method="add_topic")

    def delete_topic(self, dataset_id: int, topic: str) -> int:
        _ = (dataset_id, topic)  # unused method arg mypy error
        raise self._not_supported(method="delete_topic")

    def get_online_dataset_format(self, dataset_id: int) -> str:
        """Get the dataset format for a given dataset id from the OpenML website.

        Parameters
        ----------
        dataset_id : int
            A dataset id.

        Returns
        -------
        str
            Dataset format.
        """
        dataset_json = self._http.get(f"datasets/{dataset_id}").json()
        # build a dict from the json and get the format from the dataset description
        return dataset_json["data_set_description"]["format"].lower()  # type: ignore

    def get_online_dataset_arff(self, dataset_id: int) -> str | None:
        """Download the ARFF file for a given dataset id
        from the OpenML website.

        Parameters
        ----------
        dataset_id : int
            A dataset id.

        Returns
        -------
        str or None
            A string representation of an ARFF file. Or None if file already exists.
        """
        dataset_json = self._http.get(f"datasets/{dataset_id}").json()
        # build a dict from the json.
        # use the url from the dataset description and return the ARFF string
        arff_file = self.download_dataset_arff(dataset_json)
        with arff_file.open("r", encoding="utf8") as f:
            return f.read()
