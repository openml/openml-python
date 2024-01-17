# License: BSD 3-Clause
# ruff: noqa: PLR0913
from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload
from typing_extensions import Literal

import arff
import minio.error
import numpy as np
import pandas as pd
import urllib3
import xmltodict
from pyexpat import ExpatError
from scipy.sparse import coo_matrix

import openml._api_calls
import openml.utils
from openml.exceptions import (
    OpenMLHashException,
    OpenMLPrivateDatasetError,
    OpenMLServerError,
    OpenMLServerException,
)
from openml.utils import (
    _create_cache_directory_for_id,
    _get_cache_dir_for_id,
    _remove_cache_dir_for_id,
)

from .dataset import OpenMLDataset

if TYPE_CHECKING:
    import scipy

DATASETS_CACHE_DIR_NAME = "datasets"
logger = logging.getLogger(__name__)

NO_ACCESS_GRANTED_ERRCODE = 112

############################################################################
# Local getters/accessors to the cache directory


def _get_cache_directory(dataset: OpenMLDataset) -> Path:
    """Creates and returns the cache directory of the OpenMLDataset."""
    assert dataset.dataset_id is not None
    return _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, dataset.dataset_id)


def list_qualities() -> list[str]:
    """Return list of data qualities available.

    The function performs an API call to retrieve the entire list of
    data qualities that are computed on the datasets uploaded.

    Returns
    -------
    list
    """
    api_call = "data/qualities/list"
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    qualities = xmltodict.parse(xml_string, force_list=("oml:quality"))
    # Minimalistic check if the XML is useful
    if "oml:data_qualities_list" not in qualities:
        raise ValueError('Error in return XML, does not contain "oml:data_qualities_list"')

    if not isinstance(qualities["oml:data_qualities_list"]["oml:quality"], list):
        raise TypeError("Error in return XML, does not contain " '"oml:quality" as a list')

    return qualities["oml:data_qualities_list"]["oml:quality"]


@overload
def list_datasets(
    data_id: list[int] | None = ...,
    offset: int | None = ...,
    size: int | None = ...,
    status: str | None = ...,
    tag: str | None = ...,
    *,
    output_format: Literal["dataframe"],
    **kwargs: Any,
) -> pd.DataFrame:
    ...


@overload
def list_datasets(
    data_id: list[int] | None,
    offset: int | None,
    size: int | None,
    status: str | None,
    tag: str | None,
    output_format: Literal["dataframe"],
    **kwargs: Any,
) -> pd.DataFrame:
    ...


@overload
def list_datasets(
    data_id: list[int] | None = ...,
    offset: int | None = ...,
    size: int | None = ...,
    status: str | None = ...,
    tag: str | None = ...,
    output_format: Literal["dict"] = "dict",
    **kwargs: Any,
) -> pd.DataFrame:
    ...


def list_datasets(
    data_id: list[int] | None = None,
    offset: int | None = None,
    size: int | None = None,
    status: str | None = None,
    tag: str | None = None,
    output_format: Literal["dataframe", "dict"] = "dict",
    **kwargs: Any,
) -> dict | pd.DataFrame:
    """
    Return a list of all dataset which are on OpenML.
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
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    kwargs : dict, optional
        Legal filter operators (keys in the dict):
        data_name, data_version, number_instances,
        number_features, number_classes, number_missing_values.

    Returns
    -------
    datasets : dict of dicts, or dataframe
        - If output_format='dict'
            A mapping from dataset ID to dict.

            Every dataset is represented by a dictionary containing
            the following information:
            - dataset id
            - name
            - format
            - status
            If qualities are calculated for the dataset, some of
            these are also returned.

        - If output_format='dataframe'
            Each row maps to a dataset
            Each column contains the following information:
            - dataset id
            - name
            - format
            - status
            If qualities are calculated for the dataset, some of
            these are also included as columns.
    """
    if output_format not in ["dataframe", "dict"]:
        raise ValueError(
            "Invalid output format selected. " "Only 'dict' or 'dataframe' applicable.",
        )

    # TODO: [0.15]
    if output_format == "dict":
        msg = (
            "Support for `output_format` of 'dict' will be removed in 0.15 "
            "and pandas dataframes will be returned instead. To ensure your code "
            "will continue to work, use `output_format`='dataframe'."
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=2)

    return openml.utils._list_all(  # type: ignore
        data_id=data_id,
        list_output_format=output_format,  # type: ignore
        listing_call=_list_datasets,
        offset=offset,
        size=size,
        status=status,
        tag=tag,
        **kwargs,
    )


@overload
def _list_datasets(
    data_id: list | None = ...,
    output_format: Literal["dict"] = "dict",
    **kwargs: Any,
) -> dict:
    ...


@overload
def _list_datasets(
    data_id: list | None = ...,
    output_format: Literal["dataframe"] = "dataframe",
    **kwargs: Any,
) -> pd.DataFrame:
    ...


def _list_datasets(
    data_id: list | None = None,
    output_format: Literal["dict", "dataframe"] = "dict",
    **kwargs: Any,
) -> dict | pd.DataFrame:
    """
    Perform api call to return a list of all datasets.

    Parameters
    ----------
    The arguments that are lists are separated from the single value
    ones which are put into the kwargs.
    display_errors is also separated from the kwargs since it has a
    default value.

    data_id : list, optional

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    kwargs : dict, optional
        Legal filter operators (keys in the dict):
        tag, status, limit, offset, data_name, data_version, number_instances,
        number_features, number_classes, number_missing_values.

    Returns
    -------
    datasets : dict of dicts, or dataframe
    """
    api_call = "data/list"

    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += f"/{operator}/{value}"
    if data_id is not None:
        api_call += "/data_id/%s" % ",".join([str(int(i)) for i in data_id])
    return __list_datasets(api_call=api_call, output_format=output_format)


@overload
def __list_datasets(api_call: str, output_format: Literal["dict"] = "dict") -> dict:
    ...


@overload
def __list_datasets(api_call: str, output_format: Literal["dataframe"]) -> pd.DataFrame:
    ...


def __list_datasets(
    api_call: str,
    output_format: Literal["dict", "dataframe"] = "dict",
) -> dict | pd.DataFrame:
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
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

    if output_format == "dataframe":
        datasets = pd.DataFrame.from_dict(datasets, orient="index")

    return datasets


def _expand_parameter(parameter: str | list[str] | None) -> list[str]:
    expanded_parameter = []
    if isinstance(parameter, str):
        expanded_parameter = [x.strip() for x in parameter.split(",")]
    elif isinstance(parameter, list):
        expanded_parameter = parameter
    return expanded_parameter


def _validated_data_attributes(
    attributes: list[str],
    data_attributes: list[tuple[str, Any]],
    parameter_name: str,
) -> None:
    for attribute_ in attributes:
        is_attribute_a_data_attribute = any(dattr[0] == attribute_ for dattr in data_attributes)
        if not is_attribute_a_data_attribute:
            raise ValueError(
                f"all attribute of '{parameter_name}' should be one of the data attribute. "
                f" Got '{attribute_}' while candidates are"
                f" {[dattr[0] for dattr in data_attributes]}.",
            )


def check_datasets_active(
    dataset_ids: list[int],
    raise_error_if_not_exist: bool = True,  # noqa: FBT001, FBT002
) -> dict[int, bool]:
    """
    Check if the dataset ids provided are active.

    Raises an error if a dataset_id in the given list
    of dataset_ids does not exist on the server and
    `raise_error_if_not_exist` is set to True (default).

    Parameters
    ----------
    dataset_ids : List[int]
        A list of integers representing dataset ids.
    raise_error_if_not_exist : bool (default=True)
        Flag that if activated can raise an error, if one or more of the
        given dataset ids do not exist on the server.

    Returns
    -------
    dict
        A dictionary with items {did: bool}
    """
    datasets = list_datasets(status="all", data_id=dataset_ids, output_format="dataframe")
    missing = set(dataset_ids) - set(datasets.get("did", []))
    if raise_error_if_not_exist and missing:
        missing_str = ", ".join(str(did) for did in missing)
        raise ValueError(f"Could not find dataset(s) {missing_str} in OpenML dataset list.")
    return dict(datasets["status"] == "active")


def _name_to_id(
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
    candidates = list_datasets(
        data_name=dataset_name,
        status=status,
        data_version=version,
        output_format="dataframe",
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


def get_datasets(
    dataset_ids: list[str | int],
    download_data: bool = True,  # noqa: FBT001, FBT002
    download_qualities: bool = True,  # noqa: FBT001, FBT002
) -> list[OpenMLDataset]:
    """Download datasets.

    This function iterates :meth:`openml.datasets.get_dataset`.

    Parameters
    ----------
    dataset_ids : iterable
        Integers or strings representing dataset ids or dataset names.
        If dataset names are specified, the least recent still active dataset version is returned.
    download_data : bool, optional
        If True, also download the data file. Beware that some datasets are large and it might
        make the operation noticeably slower. Metadata is also still retrieved.
        If False, create the OpenMLDataset and only populate it with the metadata.
        The data may later be retrieved through the `OpenMLDataset.get_data` method.
    download_qualities : bool, optional (default=True)
        If True, also download qualities.xml file. If False it skip the qualities.xml.

    Returns
    -------
    datasets : list of datasets
        A list of dataset objects.
    """
    datasets = []
    for dataset_id in dataset_ids:
        datasets.append(
            get_dataset(dataset_id, download_data, download_qualities=download_qualities),
        )
    return datasets


@openml.utils.thread_safe_if_oslo_installed
def get_dataset(  # noqa: C901, PLR0912
    dataset_id: int | str,
    download_data: bool | None = None,  # Optional for deprecation warning; later again only bool
    version: int | None = None,
    error_if_multiple: bool = False,  # noqa: FBT002, FBT001
    cache_format: Literal["pickle", "feather"] = "pickle",
    download_qualities: bool | None = None,  # Same as above
    download_features_meta_data: bool | None = None,  # Same as above
    download_all_files: bool = False,  # noqa: FBT002, FBT001
    force_refresh_cache: bool = False,  # noqa: FBT001, FBT002
) -> OpenMLDataset:
    """Download the OpenML dataset representation, optionally also download actual data file.

    This function is by default NOT thread/multiprocessing safe, as this function uses caching.
    A check will be performed to determine if the information has previously been downloaded to a
    cache, and if so be loaded from disk instead of retrieved from the server.

    To make this function thread safe, you can install the python package ``oslo.concurrency``.
    If ``oslo.concurrency`` is installed `get_dataset` becomes thread safe.

    Alternatively, to make this function thread/multiprocessing safe initialize the cache first by
    calling `get_dataset(args)` once before calling `get_dataset(args)` many times in parallel.
    This will initialize the cache and later calls will use the cache in a thread/multiprocessing
    safe way.

    If dataset is retrieved by name, a version may be specified.
    If no version is specified and multiple versions of the dataset exist,
    the earliest version of the dataset that is still active will be returned.
    If no version is specified, multiple versions of the dataset exist and
    ``exception_if_multiple`` is set to ``True``, this function will raise an exception.

    Parameters
    ----------
    dataset_id : int or str
        Dataset ID of the dataset to download
    download_data : bool (default=True)
        If True, also download the data file. Beware that some datasets are large and it might
        make the operation noticeably slower. Metadata is also still retrieved.
        If False, create the OpenMLDataset and only populate it with the metadata.
        The data may later be retrieved through the `OpenMLDataset.get_data` method.
    version : int, optional (default=None)
        Specifies the version if `dataset_id` is specified by name.
        If no version is specified, retrieve the least recent still active version.
    error_if_multiple : bool (default=False)
        If ``True`` raise an error if multiple datasets are found with matching criteria.
    cache_format : str (default='pickle') in {'pickle', 'feather'}
        Format for caching the dataset - may be feather or pickle
        Note that the default 'pickle' option may load slower than feather when
        no.of.rows is very high.
    download_qualities : bool (default=True)
        Option to download 'qualities' meta-data in addition to the minimal dataset description.
        If True, download and cache the qualities file.
        If False, create the OpenMLDataset without qualities metadata. The data may later be added
        to the OpenMLDataset through the `OpenMLDataset.load_metadata(qualities=True)` method.
    download_features_meta_data : bool (default=True)
        Option to download 'features' meta-data in addition to the minimal dataset description.
        If True, download and cache the features file.
        If False, create the OpenMLDataset without features metadata. The data may later be added
        to the OpenMLDataset through the `OpenMLDataset.load_metadata(features=True)` method.
    download_all_files: bool (default=False)
        EXPERIMENTAL. Download all files related to the dataset that reside on the server.
        Useful for datasets which refer to auxiliary files (e.g., meta-album).
    force_refresh_cache : bool (default=False)
        Force the cache to refreshed by deleting the cache directory and re-downloading the data.
        Note, if `force_refresh_cache` is True, `get_dataset` is NOT thread/multiprocessing safe,
        because this creates a race condition to creating and deleting the cache; as in general with
        the cache.

    Returns
    -------
    dataset : :class:`openml.OpenMLDataset`
        The downloaded dataset.
    """
    # TODO(0.15): Remove the deprecation warning and make the default False; adjust types above
    #   and documentation. Also remove None-to-True-cases below
    if any(
        download_flag is None
        for download_flag in [download_data, download_qualities, download_features_meta_data]
    ):
        warnings.warn(
            "Starting from Version 0.15 `download_data`, `download_qualities`, and `download_featu"
            "res_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy "
            "loading. To disable this message until version 0.15 explicitly set `download_data`, "
            "`download_qualities`, and `download_features_meta_data` to a bool while calling "
            "`get_dataset`.",
            FutureWarning,
            stacklevel=2,
        )

    download_data = True if download_data is None else download_data
    download_qualities = True if download_qualities is None else download_qualities
    download_features_meta_data = (
        True if download_features_meta_data is None else download_features_meta_data
    )

    if download_all_files:
        warnings.warn(
            "``download_all_files`` is experimental and is likely to break with new releases.",
            FutureWarning,
            stacklevel=2,
        )

    if cache_format not in ["feather", "pickle"]:
        raise ValueError(
            "cache_format must be one of 'feather' or 'pickle. "
            f"Invalid format specified: {cache_format}",
        )

    if isinstance(dataset_id, str):
        try:
            dataset_id = int(dataset_id)
        except ValueError:
            dataset_id = _name_to_id(dataset_id, version, error_if_multiple)  # type: ignore
    elif not isinstance(dataset_id, int):
        raise TypeError(
            f"`dataset_id` must be one of `str` or `int`, not {type(dataset_id)}.",
        )

    if force_refresh_cache:
        did_cache_dir = _get_cache_dir_for_id(DATASETS_CACHE_DIR_NAME, dataset_id)
        if did_cache_dir.exists():
            _remove_cache_dir_for_id(DATASETS_CACHE_DIR_NAME, did_cache_dir)

    did_cache_dir = _create_cache_directory_for_id(
        DATASETS_CACHE_DIR_NAME,
        dataset_id,
    )

    remove_dataset_cache = True
    try:
        description = _get_dataset_description(did_cache_dir, dataset_id)
        features_file = None
        qualities_file = None

        if download_features_meta_data:
            features_file = _get_dataset_features_file(did_cache_dir, dataset_id)
        if download_qualities:
            qualities_file = _get_dataset_qualities_file(did_cache_dir, dataset_id)

        arff_file = _get_dataset_arff(description) if download_data else None
        if "oml:parquet_url" in description and download_data:
            try:
                parquet_file = _get_dataset_parquet(
                    description,
                    download_all_files=download_all_files,
                )
            except urllib3.exceptions.MaxRetryError:
                parquet_file = None
            if parquet_file is None and arff_file:
                logger.warning("Failed to download parquet, fallback on ARFF.")
        else:
            parquet_file = None
        remove_dataset_cache = False
    except OpenMLServerException as e:
        # if there was an exception
        # check if the user had access to the dataset
        if e.code == NO_ACCESS_GRANTED_ERRCODE:
            raise OpenMLPrivateDatasetError(e.message) from None

        raise e
    finally:
        if remove_dataset_cache:
            _remove_cache_dir_for_id(DATASETS_CACHE_DIR_NAME, did_cache_dir)

    return _create_dataset_from_description(
        description,
        features_file,
        qualities_file,
        arff_file,
        parquet_file,
        cache_format,
    )


def attributes_arff_from_df(df: pd.DataFrame) -> list[tuple[str, list[str] | str]]:
    """Describe attributes of the dataframe according to ARFF specification.

    Parameters
    ----------
    df : DataFrame, shape (n_samples, n_features)
        The dataframe containing the data set.

    Returns
    -------
    attributes_arff : list[str]
        The data set attributes as required by the ARFF format.
    """
    PD_DTYPES_TO_ARFF_DTYPE = {"integer": "INTEGER", "floating": "REAL", "string": "STRING"}
    attributes_arff: list[tuple[str, list[str] | str]] = []

    if not all(isinstance(column_name, str) for column_name in df.columns):
        logger.warning("Converting non-str column names to str.")
        df.columns = [str(column_name) for column_name in df.columns]

    for column_name in df:
        # skipna=True does not infer properly the dtype. The NA values are
        # dropped before the inference instead.
        column_dtype = pd.api.types.infer_dtype(df[column_name].dropna(), skipna=False)

        if column_dtype == "categorical":
            # for categorical feature, arff expects a list string. However, a
            # categorical column can contain mixed type and should therefore
            # raise an error asking to convert all entries to string.
            categories = df[column_name].cat.categories
            categories_dtype = pd.api.types.infer_dtype(categories)
            if categories_dtype not in ("string", "unicode"):
                raise ValueError(
                    f"The column '{column_name}' of the dataframe is of "
                    "'category' dtype. Therefore, all values in "
                    "this columns should be string. Please "
                    "convert the entries which are not string. "
                    f"Got {categories_dtype} dtype in this column.",
                )
            attributes_arff.append((column_name, categories.tolist()))
        elif column_dtype == "boolean":
            # boolean are encoded as categorical.
            attributes_arff.append((column_name, ["True", "False"]))
        elif column_dtype in PD_DTYPES_TO_ARFF_DTYPE:
            attributes_arff.append((column_name, PD_DTYPES_TO_ARFF_DTYPE[column_dtype]))
        else:
            raise ValueError(
                f"The dtype '{column_dtype}' of the column '{column_name}' is not "
                "currently supported by liac-arff. Supported "
                "dtypes are categorical, string, integer, "
                "floating, and boolean.",
            )
    return attributes_arff


def create_dataset(  # noqa: C901, PLR0912, PLR0915
    name: str,
    description: str | None,
    creator: str | None,
    contributor: str | None,
    collection_date: str | None,
    language: str | None,
    licence: str | None,
    # TODO(eddiebergman): Docstring says `type` but I don't know what this is other than strings
    # Edit: Found it could also be like ["True", "False"]
    attributes: list[tuple[str, str | list[str]]] | dict[str, str | list[str]] | Literal["auto"],
    data: pd.DataFrame | np.ndarray | scipy.sparse.coo_matrix,
    # TODO(eddiebergman): Function requires `default_target_attribute` exist but API allows None
    default_target_attribute: str,
    ignore_attribute: str | list[str] | None,
    citation: str,
    row_id_attribute: str | None = None,
    original_data_url: str | None = None,
    paper_url: str | None = None,
    update_comment: str | None = None,
    version_label: str | None = None,
) -> OpenMLDataset:
    """Create a dataset.

    This function creates an OpenMLDataset object.
    The OpenMLDataset object contains information related to the dataset
    and the actual data file.

    Parameters
    ----------
    name : str
        Name of the dataset.
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
    licence : str
        License of the data.
    attributes : list, dict, or 'auto'
        A list of tuples. Each tuple consists of the attribute name and type.
        If passing a pandas DataFrame, the attributes can be automatically
        inferred by passing ``'auto'``. Specific attributes can be manually
        specified by a passing a dictionary where the key is the name of the
        attribute and the value is the data type of the attribute.
    data : ndarray, list, dataframe, coo_matrix, shape (n_samples, n_features)
        An array that contains both the attributes and the targets. When
        providing a dataframe, the attribute names and type can be inferred by
        passing ``attributes='auto'``.
        The target feature is indicated as meta-data of the dataset.
    default_target_attribute : str
        The default target attribute, if it exists.
        Can have multiple values, comma separated.
    ignore_attribute : str | list
        Attributes that should be excluded in modelling,
        such as identifiers and indexes.
        Can have multiple values, comma separated.
    citation : str
        Reference(s) that should be cited when building on this data.
    version_label : str, optional
        Version label provided by user.
         Can be a date, hash, or some other type of id.
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
    update_comment : str, optional
        An explanation for when the dataset is uploaded.

    Returns
    -------
    class:`openml.OpenMLDataset`
    Dataset description.
    """
    if isinstance(data, pd.DataFrame):
        # infer the row id from the index of the dataset
        if row_id_attribute is None:
            row_id_attribute = data.index.name
        # When calling data.values, the index will be skipped.
        # We need to reset the index such that it is part of the data.
        if data.index.name is not None:
            data = data.reset_index()

    if attributes == "auto" or isinstance(attributes, dict):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Automatically inferring attributes requires "
                f"a pandas DataFrame. A {data!r} was given instead.",
            )
        # infer the type of data for each column of the DataFrame
        attributes_ = attributes_arff_from_df(data)
        if isinstance(attributes, dict):
            # override the attributes which was specified by the user
            for attr_idx in range(len(attributes_)):
                attr_name = attributes_[attr_idx][0]
                if attr_name in attributes:
                    attributes_[attr_idx] = (attr_name, attributes[attr_name])
    else:
        attributes_ = attributes
    ignore_attributes = _expand_parameter(ignore_attribute)
    _validated_data_attributes(ignore_attributes, attributes_, "ignore_attribute")

    default_target_attributes = _expand_parameter(default_target_attribute)
    _validated_data_attributes(default_target_attributes, attributes_, "default_target_attribute")

    if row_id_attribute is not None:
        is_row_id_an_attribute = any(attr[0] == row_id_attribute for attr in attributes_)
        if not is_row_id_an_attribute:
            raise ValueError(
                "'row_id_attribute' should be one of the data attribute. "
                " Got '{}' while candidates are {}.".format(
                    row_id_attribute,
                    [attr[0] for attr in attributes_],
                ),
            )

    if isinstance(data, pd.DataFrame):
        if all(isinstance(dtype, pd.SparseDtype) for dtype in data.dtypes):
            data = data.sparse.to_coo()
            # liac-arff only support COO matrices with sorted rows
            row_idx_sorted = np.argsort(data.row)  # type: ignore
            data.row = data.row[row_idx_sorted]  # type: ignore
            data.col = data.col[row_idx_sorted]  # type: ignore
            data.data = data.data[row_idx_sorted]  # type: ignore
        else:
            data = data.to_numpy()

    data_format: Literal["arff", "sparse_arff"]
    if isinstance(data, (list, np.ndarray)):
        if isinstance(data[0], (list, np.ndarray)):
            data_format = "arff"
        elif isinstance(data[0], dict):
            data_format = "sparse_arff"
        else:
            raise ValueError(
                "When giving a list or a numpy.ndarray, "
                "they should contain a list/ numpy.ndarray "
                "for dense data or a dictionary for sparse "
                f"data. Got {data[0]!r} instead.",
            )
    elif isinstance(data, coo_matrix):
        data_format = "sparse_arff"
    else:
        raise ValueError(
            "When giving a list or a numpy.ndarray, "
            "they should contain a list/ numpy.ndarray "
            "for dense data or a dictionary for sparse "
            f"data. Got {data[0]!r} instead.",
        )

    arff_object = {
        "relation": name,
        "description": description,
        "attributes": attributes_,
        "data": data,
    }

    # serializes the ARFF dataset object and returns a string
    arff_dataset = arff.dumps(arff_object)
    try:
        # check if ARFF is valid
        decoder = arff.ArffDecoder()
        return_type = arff.COO if data_format == "sparse_arff" else arff.DENSE
        decoder.decode(arff_dataset, encode_nominal=True, return_type=return_type)
    except arff.ArffException as e:
        raise ValueError(
            "The arguments you have provided do not construct a valid ARFF file"
        ) from e

    return OpenMLDataset(
        name=name,
        description=description,
        data_format=data_format,
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
        update_comment=update_comment,
        dataset=arff_dataset,
    )


def status_update(data_id: int, status: Literal["active", "deactivated"]) -> None:
    """
    Updates the status of a dataset to either 'active' or 'deactivated'.
    Please see the OpenML API documentation for a description of the status
    and all legal status transitions:
    https://docs.openml.org/#dataset-status

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
    result_xml = openml._api_calls._perform_api_call("data/status/update", "post", data=data)
    result = xmltodict.parse(result_xml)
    server_data_id = result["oml:data_status_update"]["oml:id"]
    server_status = result["oml:data_status_update"]["oml:status"]
    if status != server_status or int(data_id) != int(server_data_id):
        # This should never happen
        raise ValueError("Data id/status does not collide")


def edit_dataset(
    data_id: int,
    description: str | None = None,
    creator: str | None = None,
    contributor: str | None = None,
    collection_date: str | None = None,
    language: str | None = None,
    default_target_attribute: str | None = None,
    ignore_attribute: str | list[str] | None = None,
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
    result_xml = openml._api_calls._perform_api_call(
        "data/edit",
        "post",
        data=form_data,
        file_elements=file_elements,
    )
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_edit"]["oml:id"]
    return int(data_id)


def fork_dataset(data_id: int) -> int:
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
    form_data = {"data_id": data_id}  # type: openml._api_calls.DATA_TYPE
    result_xml = openml._api_calls._perform_api_call("data/fork", "post", data=form_data)
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_fork"]["oml:id"]
    return int(data_id)


def data_feature_add_ontology(data_id: int, index: int, ontology: str) -> bool:
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
    upload_data: dict[str, int | str] = {"data_id": data_id, "index": index, "ontology": ontology}
    openml._api_calls._perform_api_call("data/feature/ontology/add", "post", data=upload_data)
    # an error will be thrown in case the request was unsuccessful
    return True


def data_feature_remove_ontology(data_id: int, index: int, ontology: str) -> bool:
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
    upload_data: dict[str, int | str] = {"data_id": data_id, "index": index, "ontology": ontology}
    openml._api_calls._perform_api_call("data/feature/ontology/remove", "post", data=upload_data)
    # an error will be thrown in case the request was unsuccessful
    return True


def _topic_add_dataset(data_id: int, topic: str) -> int:
    """
    Adds a topic for a dataset.
    This API is not available for all OpenML users and is accessible only by admins.

    Parameters
    ----------
    data_id : int
        id of the dataset for which the topic needs to be added
    topic : str
        Topic to be added for the dataset

    Returns
    -------
    Dataset id
    """
    if not isinstance(data_id, int):
        raise TypeError(f"`data_id` must be of type `int`, not {type(data_id)}.")
    form_data = {"data_id": data_id, "topic": topic}  # type: openml._api_calls.DATA_TYPE
    result_xml = openml._api_calls._perform_api_call("data/topicadd", "post", data=form_data)
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_topic"]["oml:id"]
    return int(data_id)


def _topic_delete_dataset(data_id: int, topic: str) -> int:
    """
    Removes a topic from a dataset.
    This API is not available for all OpenML users and is accessible only by admins.

    Parameters
    ----------
    data_id : int
        id of the dataset to be forked
    topic : str
        Topic to be deleted

    Returns
    -------
    Dataset id
    """
    if not isinstance(data_id, int):
        raise TypeError(f"`data_id` must be of type `int`, not {type(data_id)}.")
    form_data = {"data_id": data_id, "topic": topic}  # type: openml._api_calls.DATA_TYPE
    result_xml = openml._api_calls._perform_api_call("data/topicdelete", "post", data=form_data)
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_topic"]["oml:id"]
    return int(data_id)


def _get_dataset_description(did_cache_dir: Path, dataset_id: int) -> dict[str, Any]:
    """Get the dataset description as xml dictionary.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : Path
        Cache subdirectory for this dataset.

    dataset_id : int
        Dataset ID

    Returns
    -------
    dict
        XML Dataset description parsed to a dict.

    """
    # TODO implement a cache for this that invalidates itself after some time
    # This can be saved on disk, but cannot be cached properly, because
    # it contains the information on whether a dataset is active.
    description_file = did_cache_dir / "description.xml"

    try:
        with description_file.open(encoding="utf8") as fh:
            dataset_xml = fh.read()
        description = xmltodict.parse(dataset_xml)["oml:data_set_description"]
    except Exception:  # noqa: BLE001
        url_extension = f"data/{dataset_id}"
        dataset_xml = openml._api_calls._perform_api_call(url_extension, "get")
        try:
            description = xmltodict.parse(dataset_xml)["oml:data_set_description"]
        except ExpatError as e:
            url = openml._api_calls._create_url_from_endpoint(url_extension)
            raise OpenMLServerError(f"Dataset description XML at '{url}' is malformed.") from e

        with description_file.open("w", encoding="utf8") as fh:
            fh.write(dataset_xml)

    return description  # type: ignore


def _get_dataset_parquet(
    description: dict | OpenMLDataset,
    cache_directory: Path | None = None,
    download_all_files: bool = False,  # noqa: FBT001, FBT002
) -> Path | None:
    """Return the path to the local parquet file of the dataset. If is not cached, it is downloaded.

    Checks if the file is in the cache, if yes, return the path to the file.
    If not, downloads the file and caches it, then returns the file path.
    The cache directory is generated based on dataset information, but can also be specified.

    This function is NOT thread/multiprocessing safe.
    Unlike the ARFF equivalent, checksums are not available/used (for now).

    Parameters
    ----------
    description : dictionary or OpenMLDataset
        Either a dataset description as dict or OpenMLDataset.

    cache_directory: Path, optional (default=None)
        Folder to store the parquet file in.
        If None, use the default cache directory for the dataset.

    download_all_files: bool, optional (default=False)
        If `True`, download all data found in the bucket to which the description's
        ``parquet_url`` points, only download the parquet file otherwise.

    Returns
    -------
    output_filename : Path, optional
        Location of the Parquet file if successfully downloaded, None otherwise.
    """
    if isinstance(description, dict):
        url = str(description.get("oml:parquet_url"))
        did = int(description.get("oml:id"))  # type: ignore
    elif isinstance(description, OpenMLDataset):
        url = str(description._parquet_url)
        assert description.dataset_id is not None

        did = int(description.dataset_id)
    else:
        raise TypeError("`description` should be either OpenMLDataset or Dict.")

    if cache_directory is None:
        cache_directory = _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, did)

    output_file_path = cache_directory / f"dataset_{did}.pq"

    old_file_path = cache_directory / "dataset.pq"
    if old_file_path.is_file():
        old_file_path.rename(output_file_path)

    # For this release, we want to be able to force a new download even if the
    # parquet file is already present when ``download_all_files`` is set.
    # For now, it would be the only way for the user to fetch the additional
    # files in the bucket (no function exists on an OpenMLDataset to do this).
    if download_all_files:
        openml._api_calls._download_minio_bucket(source=url, destination=cache_directory)

    if not output_file_path.is_file():
        try:
            openml._api_calls._download_minio_file(
                source=url,
                destination=output_file_path,
            )
        except (FileNotFoundError, urllib3.exceptions.MaxRetryError, minio.error.ServerError) as e:
            logger.warning(f"Could not download file from {url}: {e}")
            return None
    return output_file_path


def _get_dataset_arff(
    description: dict | OpenMLDataset,
    cache_directory: Path | None = None,
) -> Path:
    """Return the path to the local arff file of the dataset. If is not cached, it is downloaded.

    Checks if the file is in the cache, if yes, return the path to the file.
    If not, downloads the file and caches it, then returns the file path.
    The cache directory is generated based on dataset information, but can also be specified.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    description : dictionary or OpenMLDataset
        Either a dataset description as dict or OpenMLDataset.

    cache_directory: Path, optional (default=None)
        Folder to store the arff file in.
        If None, use the default cache directory for the dataset.

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

    save_cache_directory = (
        _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, did)
        if cache_directory is None
        else Path(cache_directory)
    )
    output_file_path = save_cache_directory / "dataset.arff"

    try:
        openml._api_calls._download_text_file(
            source=url,
            output_path=output_file_path,
            md5_checksum=md5_checksum_fixture,
        )
    except OpenMLHashException as e:
        additional_info = f" Raised when downloading dataset {did}."
        e.args = (e.args[0] + additional_info,)
        raise e

    return output_file_path


def _get_features_xml(dataset_id: int) -> str:
    url_extension = f"data/features/{dataset_id}"
    return openml._api_calls._perform_api_call(url_extension, "get")


def _get_dataset_features_file(did_cache_dir: str | Path | None, dataset_id: int) -> Path:
    """API call to load dataset features. Loads from cache or downloads them.

    Features are feature descriptions for each column.
    (name, index, categorical, ...)

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str or None
        Cache subdirectory for this dataset

    dataset_id : int
        Dataset ID

    Returns
    -------
    Path
        Path of the cached dataset feature file
    """
    did_cache_dir = Path(did_cache_dir) if did_cache_dir is not None else None
    if did_cache_dir is None:
        did_cache_dir = _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, dataset_id)

    features_file = did_cache_dir / "features.xml"

    # Dataset features aren't subject to change...
    if not features_file.is_file():
        features_xml = _get_features_xml(dataset_id)
        with features_file.open("w", encoding="utf8") as fh:
            fh.write(features_xml)

    return features_file


def _get_qualities_xml(dataset_id: int) -> str:
    url_extension = f"data/qualities/{dataset_id!s}"
    return openml._api_calls._perform_api_call(url_extension, "get")


def _get_dataset_qualities_file(
    did_cache_dir: str | Path | None,
    dataset_id: int,
) -> Path | None:
    """Get the path for the dataset qualities file, or None if no qualities exist.

    Loads from cache or downloads them.
    Features are metafeatures (number of features, number of classes, ...)

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str or None
        Cache subdirectory for this dataset

    dataset_id : int
        Dataset ID

    Returns
    -------
    str
        Path of the cached qualities file
    """
    save_did_cache_dir = (
        _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, dataset_id)
        if did_cache_dir is None
        else Path(did_cache_dir)
    )

    # Dataset qualities are subject to change and must be fetched every time
    qualities_file = save_did_cache_dir / "qualities.xml"
    try:
        with qualities_file.open(encoding="utf8") as fh:
            qualities_xml = fh.read()
    except OSError:
        try:
            qualities_xml = _get_qualities_xml(dataset_id)
            with qualities_file.open("w", encoding="utf8") as fh:
                fh.write(qualities_xml)
        except OpenMLServerException as e:
            if e.code == 362 and str(e) == "No qualities found - None":
                # quality file stays as None
                logger.warning(f"No qualities found for dataset {dataset_id}")
                return None

            raise e

    return qualities_file


def _create_dataset_from_description(
    description: dict[str, str],
    features_file: Path | None = None,
    qualities_file: Path | None = None,
    arff_file: Path | None = None,
    parquet_file: Path | None = None,
    cache_format: Literal["pickle", "feather"] = "pickle",
) -> OpenMLDataset:
    """Create a dataset object from a description dict.

    Parameters
    ----------
    description : dict
        Description of a dataset in xml dict.
    features_file : str
        Path of the dataset features as xml file.
    qualities_file : list
        Path of the dataset qualities as xml file.
    arff_file : string, optional
        Path of dataset ARFF file.
    parquet_file : string, optional
        Path of dataset Parquet file.
    cache_format: string, optional
        Caching option for datasets (feather/pickle)

    Returns
    -------
    dataset : dataset object
        Dataset object from dict and ARFF.
    """
    return OpenMLDataset(
        description["oml:name"],
        description.get("oml:description"),
        data_format=description["oml:format"],  # type: ignore
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
        cache_format=cache_format,
        features_file=str(features_file) if features_file is not None else None,
        qualities_file=str(qualities_file) if qualities_file is not None else None,
        parquet_url=description.get("oml:parquet_url"),
        parquet_file=str(parquet_file) if parquet_file is not None else None,
    )


def _get_online_dataset_arff(dataset_id: int) -> str | None:
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
    dataset_xml = openml._api_calls._perform_api_call("data/%d" % dataset_id, "get")
    # build a dict from the xml.
    # use the url from the dataset description and return the ARFF string
    return openml._api_calls._download_text_file(
        xmltodict.parse(dataset_xml)["oml:data_set_description"]["oml:url"],
    )


def _get_online_dataset_format(dataset_id: int) -> str:
    """Get the dataset format for a given dataset id
    from the OpenML website.

    Parameters
    ----------
    dataset_id : int
        A dataset id.

    Returns
    -------
    str
        Dataset format.
    """
    dataset_xml = openml._api_calls._perform_api_call("data/%d" % dataset_id, "get")
    # build a dict from the xml and get the format from the dataset description
    return xmltodict.parse(dataset_xml)["oml:data_set_description"]["oml:format"].lower()  # type: ignore


def delete_dataset(dataset_id: int) -> bool:
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
