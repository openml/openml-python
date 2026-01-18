from __future__ import annotations

import builtins
import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from openml._api.resources.base import DatasetsAPI
from openml.datasets.data_feature import OpenMLDataFeature
from openml.datasets.dataset import OpenMLDataset
from openml.exceptions import OpenMLNotAuthorizedError, OpenMLServerError, OpenMLServerException

if TYPE_CHECKING:
    from requests import Response

    import openml

import pandas as pd
import xmltodict

logger = logging.getLogger(__name__)


NO_ACCESS_GRANTED_ERRCODE = 112


class DatasetsV1(DatasetsAPI):
    def get(
        self,
        dataset_id: int | str,
        *,
        return_response: bool = False,
    ) -> OpenMLDataset | tuple[OpenMLDataset, Response]:
        path = f"data/{dataset_id}"
        response = self._http.get(path)
        xml_content = response.text
        dataset = self._create_dataset_from_xml(xml_content)
        if return_response:
            return dataset, response

        return dataset

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

    def delete(self, dataset_id: int) -> bool:
        """Delete dataset with id `dataset_id` from the OpenML server.

        This can only be done if you are the owner of the dataset and
        no tasks are attached to the dataset.

        Parameters
        ----------
        dataset_id : int
            OpenML id of the dataset

        Returns
        -------
        bool
            True if the deletion was successful. False otherwise.
        """
        url_suffix = f"data/{dataset_id}"
        try:
            result_xml = self._http.delete(url_suffix)
            result = xmltodict.parse(result_xml)
            return "oml:data_delete" in result
        except OpenMLServerException as e:
            # https://github.com/openml/OpenML/blob/21f6188d08ac24fcd2df06ab94cf421c946971b0/openml_OS/views/pages/api_new/v1/xml/pre.php
            # Most exceptions are descriptive enough to be raised as their standard
            # OpenMLServerException, however there are two cases where we add information:
            #  - a generic "failed" message, we direct them to the right issue board
            #  - when the user successfully authenticates with the server,
            #    but user is not allowed to take the requested action,
            #    in which case we specify a OpenMLNotAuthorizedError.
            by_other_user = [323, 353, 393, 453, 594]
            has_dependent_entities = [324, 326, 327, 328, 354, 454, 464, 595]
            unknown_reason = [325, 355, 394, 455, 593]
            if e.code in by_other_user:
                raise OpenMLNotAuthorizedError(
                    message=("The data can not be deleted because it was not uploaded by you."),
                ) from e
            if e.code in has_dependent_entities:
                raise OpenMLNotAuthorizedError(
                    message=(
                        f"The data can not be deleted because "
                        f"it still has associated entities: {e.message}"
                    ),
                ) from e
            if e.code in unknown_reason:
                raise OpenMLServerError(
                    message=(
                        "The data can not be deleted for unknown reason,"
                        " please open an issue at: https://github.com/openml/openml/issues/new"
                    ),
                ) from e
            raise e

    def edit(  # noqa: PLR0913
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

        In addition to providing the dataset id of the dataset to edit (through dataset_id),
        you must specify a value for at least one of the optional function arguments,
        i.e. one value for a field to edit.

        This function allows editing of both non-critical and critical fields.
        Critical fields are default_target_attribute, ignore_attribute, row_id_attribute.

        - Editing non-critical data fields is allowed for all authenticated users.
        - Editing critical fields is allowed only for the owner, provided there are no tasks
        associated with this dataset.

        If dataset has tasks or if the user is not the owner, the only way
        to edit critical fields is to use fork_dataset followed by edit_dataset.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset.
        description : str
            Description of the dataset.
        creator : str
            The person who created the dataset.
        contributor : str
            People who contributed to the current version of the dataset.
        collection_date : str
            The date the data was originally collected, given by the uploader.
        language : str
            Language in which the data is represented.
            Starts with 1 upper case letter, rest lower case, e.g. 'English'.
        default_target_attribute : str
            The default target attribute, if it exists.
            Can have multiple values, comma separated.
        ignore_attribute : str | list
            Attributes that should be excluded in modelling,
            such as identifiers and indexes.
        citation : str
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
        if not isinstance(dataset_id, int):
            raise TypeError(f"`dataset_id` must be of type `int`, not {type(dataset_id)}.")

        # compose data edit parameters as xml
        form_data = {"data_id": dataset_id}  # type: openml._api_calls.DATA_TYPE
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
        }  # type: openml._api_calls.FILE_ELEMENTS_TYPE
        result_xml = self._http.post("data/edit", data=form_data, files=file_elements).text
        result = xmltodict.parse(result_xml)
        dataset_id = result["oml:data_edit"]["oml:id"]
        return int(dataset_id)

    def fork(self, dataset_id: int) -> int:
        """
        Creates a new dataset version, with the authenticated user as the new owner.
        The forked dataset can have distinct dataset meta-data,
        but the actual data itself is shared with the original version.

        This API is intended for use when a user is unable to edit the critical fields of a dataset
        through the edit_dataset API.
        (Critical fields are default_target_attribute, ignore_attribute, row_id_attribute.)

        Specifically, this happens when the user is:
                1. Not the owner of the dataset.
                2. User is the owner of the dataset, but the dataset has tasks.

        In these two cases the only way to edit critical fields is:
                1. STEP 1: Fork the dataset using fork_dataset API
                2. STEP 2: Call edit_dataset API on the forked version.


        Parameters
        ----------
        dataset_id : int
            id of the dataset to be forked

        Returns
        -------
        Dataset id of the forked dataset

        """
        if not isinstance(dataset_id, int):
            raise TypeError(f"`dataset_id` must be of type `int`, not {type(dataset_id)}.")
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

        data: openml._api_calls.DATA_TYPE = {"data_id": dataset_id, "status": status}
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
        path = f"data/features/{dataset_id}"
        xml = self._http.get(path, use_cache=True).text

        return self._parse_features_xml(xml)

    def get_qualities(self, dataset_id: int) -> dict[str, float] | None:
        path = f"data/qualities/{dataset_id!s}"
        try:
            xml = self._http.get(path, use_cache=True).text
        except OpenMLServerException as e:
            if e.code == 362 and str(e) == "No qualities found - None":
                # quality file stays as None
                logger.warning(f"No qualities found for dataset {dataset_id}")
                return None

            raise e

        return self._parse_qualities_xml(xml)

    def parse_features_file(
        self, features_file: Path, features_pickle_file: Path
    ) -> dict[int, OpenMLDataFeature]:
        if features_file.suffix != ".xml":
            # TODO (Shrivaths) can only parse xml warn/ raise exception
            raise NotImplementedError()

        with Path(features_file).open("r", encoding="utf8") as fh:
            features_xml = fh.read()

        features = self._parse_features_xml(features_xml)

        with features_pickle_file.open("wb") as fh_binary:
            pickle.dump(features, fh_binary)

        return features

    def parse_qualities_file(
        self, qualities_file: Path, qualities_pickle_file: Path
    ) -> dict[str, float]:
        if qualities_file.suffix != ".xml":
            # TODO (Shrivaths) can only parse xml warn/ raise exception
            raise NotImplementedError()

        with Path(qualities_file).open("r", encoding="utf8") as fh:
            qualities_xml = fh.read()

        qualities = self._parse_qualities_xml(qualities_xml)

        with qualities_pickle_file.open("wb") as fh_binary:
            pickle.dump(qualities, fh_binary)

        return qualities

    def _parse_features_xml(self, features_xml_string: str) -> dict[int, OpenMLDataFeature]:
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

    def download_file(self, url_ext: str, encoding: str = "utf-8") -> Path:
        def __handler(response: Response, path: Path, encoding: str) -> Path:
            file_path = path / "response.xml"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding=encoding) as f:
                f.write(response.text)
            return file_path

        return self._http.download(url_ext, __handler, encoding)

    def download_features_file(self, dataset_id: int) -> Path:
        path = f"data/features/{dataset_id}"
        return self.download_file(path)

    def download_qualities_file(self, dataset_id: int) -> Path:
        path = f"data/qualities/{dataset_id}"
        return self.download_file(path)


class DatasetsV2(DatasetsAPI):
    def get(
        self,
        dataset_id: int | str,
        *,
        return_response: bool = False,
    ) -> OpenMLDataset | tuple[OpenMLDataset, Response]:
        path = f"data/{dataset_id}"
        response = self._http.get(path)
        json_content = response.json()
        dataset = self._create_dataset_from_json(json_content)

        if return_response:
            return dataset, response

        return dataset

    def list(
        self,
        limit: int,
        offset: int,
        *,
        dataset_id: builtins.list[int] | None = None,
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
        dataset_id: list[int], optional

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
        if dataset_id is not None:
            json["data_id"] = dataset_id
        if kwargs is not None:
            for operator, value in kwargs.items():
                if value is not None:
                    json[operator] = value

        api_call = "datasets/list"
        datasets_list = self._http.post(api_call, json=json).json()
        # Minimalistic check if the JSON is useful
        assert isinstance(datasets_list, list), type(datasets_list)

        return self._parse_list_json(datasets_list)

    def delete(self, dataset_id: int) -> bool:
        raise NotImplementedError()

    def edit(  # noqa: PLR0913
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
        raise NotImplementedError()

    def fork(self, dataset_id: int) -> int:
        raise NotImplementedError()

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

        data: openml._api_calls.DATA_TYPE = {"dataset_id": dataset_id, "status": status}
        result = self._http.post("datasets/status/update", json=data).json()
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

    def feature_add_ontology(self, dataset_id: int, index: int, ontology: str) -> bool:
        raise NotImplementedError()

    def feature_remove_ontology(self, dataset_id: int, index: int, ontology: str) -> bool:
        raise NotImplementedError()

    def get_features(self, dataset_id: int) -> dict[int, OpenMLDataFeature]:
        path = f"datasets/features/{dataset_id}"
        json = self._http.get(path, use_cache=True).json()

        return self._parse_features_json(json)

    def get_qualities(self, dataset_id: int) -> dict[str, float] | None:
        path = f"datasets/qualities/{dataset_id!s}"
        try:
            qualities_json = self._http.get(path, use_cache=True).json()
        except OpenMLServerException as e:
            if e.code == 362 and str(e) == "No qualities found - None":
                logger.warning(f"No qualities found for dataset {dataset_id}")
                return None

            raise e

        return self._parse_qualities_json(qualities_json)

    def parse_features_file(
        self, features_file: Path, features_pickle_file: Path
    ) -> dict[int, OpenMLDataFeature]:
        if features_file.suffix != ".json":
            # can fallback to v1 if the file is .xml
            raise NotImplementedError()

        with Path(features_file).open("r", encoding="utf8") as fh:
            features_json = json.load(fh)

        features = self._parse_features_json(features_json)

        with features_pickle_file.open("wb") as fh_binary:
            pickle.dump(features, fh_binary)

        return features

    def parse_qualities_file(
        self, qualities_file: Path, qualities_pickle_file: Path
    ) -> dict[str, float]:
        if qualities_file.suffix != ".json":
            # can fallback to v1 if the file is .xml
            raise NotImplementedError()

        with Path(qualities_file).open("r", encoding="utf8") as fh:
            qualities_json = json.load(fh)

        qualities = self._parse_qualities_json(qualities_json)

        with qualities_pickle_file.open("wb") as fh_binary:
            pickle.dump(qualities, fh_binary)

        return qualities

    def _parse_features_json(self, features_json: dict) -> dict[int, OpenMLDataFeature]:
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
        datasets = {}
        for dataset_ in datasets_list:
            ignore_attribute = ["file_id", "quality"]
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

    def download_file(self, url_ext: str, encoding: str = "utf-8") -> Path:
        def __handler(response: Response, path: Path, encoding: str) -> Path:
            file_path = path / "response.json"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding=encoding) as f:
                json.dump(response.json(), f, indent=4)
            return file_path

        return self._http.download(url_ext, __handler, encoding)

    def download_features_file(self, dataset_id: int) -> Path:
        path = f"datasets/features/{dataset_id}"
        return self.download_file(path)

    def download_qualities_file(self, dataset_id: int) -> Path:
        path = f"datasets/qualities/{dataset_id}"
        return self.download_file(path)
