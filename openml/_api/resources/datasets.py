from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from typing_extensions import Literal

if TYPE_CHECKING:
    from requests import Response

import pandas as pd
import xmltodict

import openml.utils
from openml._api.resources.base import DatasetsAPI
from openml.datasets.dataset import OpenMLDataset


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
        return openml.utils._delete_entity("data", dataset_id)

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
    ) -> int:
        """Edits an OpenMLDataset.

        In addition to providing the dataset id of the dataset to edit (through data_id),
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
        data_id : int
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
        if not isinstance(data_id, int):
            raise TypeError(f"`data_id` must be of type `int`, not {type(data_id)}.")

        # compose data edit parameters as xml
        form_data = {"data_id": data_id}  # type: openml._api_calls.DATA_TYPE
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
        data_id = result["oml:data_edit"]["oml:id"]
        return int(data_id)

    def fork(self, data_id: int) -> int:
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
        data_id : int
            id of the dataset to be forked

        Returns
        -------
        Dataset id of the forked dataset

        """
        if not isinstance(data_id, int):
            raise TypeError(f"`data_id` must be of type `int`, not {type(data_id)}.")
        # compose data fork parameters
        form_data = {"data_id": data_id}
        result_xml = self._http.post("data/fork", data=form_data).text
        result = xmltodict.parse(result_xml)
        data_id = result["oml:data_fork"]["oml:id"]
        return int(data_id)

    def status_update(self, data_id: int, status: Literal["active", "deactivated"]) -> None:
        """
        Updates the status of a dataset to either 'active' or 'deactivated'.
        Please see the OpenML API documentation for a description of the status
        and all legal status transitions:
        https://docs.openml.org/concepts/data/#dataset-status

        Parameters
        ----------
        data_id : int
            The data id of the dataset
        status : str,
            'active' or 'deactivated'
        """
        legal_status = {"active", "deactivated"}
        if status not in legal_status:
            raise ValueError(f"Illegal status value. Legal values: {legal_status}")

        data: openml._api_calls.DATA_TYPE = {"data_id": data_id, "status": status}
        result_xml = self._http.post("data/status/update", data=data).text
        result = xmltodict.parse(result_xml)
        server_data_id = result["oml:data_status_update"]["oml:id"]
        server_status = result["oml:data_status_update"]["oml:status"]
        if status != server_status or int(data_id) != int(server_data_id):
            # This should never happen
            raise ValueError("Data id/status does not collide")

    def list_qualities(self) -> list[str]:  # type: ignore
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

    def feature_add_ontology(self, data_id: int, index: int, ontology: str) -> bool:
        """
        An ontology describes the concept that are described in a feature. An
        ontology is defined by an URL where the information is provided. Adds
        an ontology (URL) to a given dataset feature (defined by a dataset id
        and index). The dataset has to exists on OpenML and needs to have been
        processed by the evaluation engine.

        Parameters
        ----------
        data_id : int
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
            "data_id": data_id,
            "index": index,
            "ontology": ontology,
        }
        self._http.post("data/feature/ontology/add", data=upload_data)
        # an error will be thrown in case the request was unsuccessful
        return True

    def feature_remove_ontology(self, data_id: int, index: int, ontology: str) -> bool:
        """
        Removes an existing ontology (URL) from a given dataset feature (defined
        by a dataset id and index). The dataset has to exists on OpenML and needs
        to have been processed by the evaluation engine. Ontology needs to be
        attached to the specific fearure.

        Parameters
        ----------
        data_id : int
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
            "data_id": data_id,
            "index": index,
            "ontology": ontology,
        }
        self._http.post("data/feature/ontology/remove", data=upload_data)
        # an error will be thrown in case the request was unsuccessful
        return True


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

    def delete(self, dataset_id: int) -> bool:
        raise NotImplementedError()

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
    ) -> int:
        raise NotImplementedError()

    def fork(self, data_id: int) -> int:
        raise NotImplementedError()

    def status_update(self, data_id: int, status: Literal["active", "deactivated"]) -> None:
        """
        Updates the status of a dataset to either 'active' or 'deactivated'.
        Please see the OpenML API documentation for a description of the status
        and all legal status transitions:
        https://docs.openml.org/concepts/data/#dataset-status

        Parameters
        ----------
        data_id : int
            The data id of the dataset
        status : str,
            'active' or 'deactivated'
        """
        legal_status = {"active", "deactivated"}
        if status not in legal_status:
            raise ValueError(f"Illegal status value. Legal values: {legal_status}")

        data: openml._api_calls.DATA_TYPE = {"dataset_id": data_id, "status": status}
        result = self._http.post("datasets/status/update", json=data).json()
        server_data_id = result["dataset_id"]
        server_status = result["status"]
        if status != server_status or int(data_id) != int(server_data_id):
            # This should never happen
            raise ValueError("Data id/status does not collide")

    def list_qualities(self) -> list[str]:  # type: ignore
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

    def feature_add_ontology(self, data_id: int, index: int, ontology: str) -> bool:
        raise NotImplementedError()

    def feature_remove_ontology(self, data_id: int, index: int, ontology: str) -> bool:
        raise NotImplementedError()
