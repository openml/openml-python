# License: BSD 3-Clause

import io
import logging
import os
from typing import List, Dict, Union, Optional, cast

import numpy as np
import arff
import pandas as pd
import urllib3

import xmltodict
from scipy.sparse import coo_matrix
from collections import OrderedDict

import openml.utils
import openml._api_calls
from .dataset import OpenMLDataset
from ..exceptions import (
    OpenMLHashException,
    OpenMLServerException,
    OpenMLPrivateDatasetError,
)
from ..utils import (
    _remove_cache_dir_for_id,
    _create_cache_directory_for_id,
)


DATASETS_CACHE_DIR_NAME = "datasets"
logger = logging.getLogger(__name__)

############################################################################
# Local getters/accessors to the cache directory


def _get_cache_directory(dataset: OpenMLDataset) -> str:
    """ Return the cache directory of the OpenMLDataset """
    return _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, dataset.dataset_id)


def list_qualities() -> List[str]:
    """ Return list of data qualities available.

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
        raise ValueError("Error in return XML, does not contain " '"oml:data_qualities_list"')
    if not isinstance(qualities["oml:data_qualities_list"]["oml:quality"], list):
        raise TypeError("Error in return XML, does not contain " '"oml:quality" as a list')
    qualities = qualities["oml:data_qualities_list"]["oml:quality"]
    return qualities


def list_datasets(
    data_id: Optional[List[int]] = None,
    offset: Optional[int] = None,
    size: Optional[int] = None,
    status: Optional[str] = None,
    tag: Optional[str] = None,
    output_format: str = "dict",
    **kwargs,
) -> Union[Dict, pd.DataFrame]:

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
            "Invalid output format selected. " "Only 'dict' or 'dataframe' applicable."
        )

    return openml.utils._list_all(
        data_id=data_id,
        output_format=output_format,
        listing_call=_list_datasets,
        offset=offset,
        size=size,
        status=status,
        tag=tag,
        **kwargs,
    )


def _list_datasets(data_id: Optional[List] = None, output_format="dict", **kwargs):
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
            api_call += "/%s/%s" % (operator, value)
    if data_id is not None:
        api_call += "/data_id/%s" % ",".join([str(int(i)) for i in data_id])
    return __list_datasets(api_call=api_call, output_format=output_format)


def __list_datasets(api_call, output_format="dict"):

    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    datasets_dict = xmltodict.parse(xml_string, force_list=("oml:dataset",))

    # Minimalistic check if the XML is useful
    assert type(datasets_dict["oml:data"]["oml:dataset"]) == list, type(datasets_dict["oml:data"])
    assert datasets_dict["oml:data"]["@xmlns:oml"] == "http://openml.org/openml", datasets_dict[
        "oml:data"
    ]["@xmlns:oml"]

    datasets = dict()
    for dataset_ in datasets_dict["oml:data"]["oml:dataset"]:
        ignore_attribute = ["oml:file_id", "oml:quality"]
        dataset = {
            k.replace("oml:", ""): v for (k, v) in dataset_.items() if k not in ignore_attribute
        }
        dataset["did"] = int(dataset["did"])
        dataset["version"] = int(dataset["version"])

        # The number of qualities can range from 0 to infinity
        for quality in dataset_.get("oml:quality", list()):
            try:
                dataset[quality["@name"]] = int(quality["#text"])
            except ValueError:
                dataset[quality["@name"]] = float(quality["#text"])
        datasets[dataset["did"]] = dataset

    if output_format == "dataframe":
        datasets = pd.DataFrame.from_dict(datasets, orient="index")

    return datasets


def _expand_parameter(parameter: Union[str, List[str]]) -> List[str]:
    expanded_parameter = []
    if isinstance(parameter, str):
        expanded_parameter = [x.strip() for x in parameter.split(",")]
    elif isinstance(parameter, list):
        expanded_parameter = parameter
    return expanded_parameter


def _validated_data_attributes(
    attributes: List[str], data_attributes: List[str], parameter_name: str
) -> None:
    for attribute_ in attributes:
        is_attribute_a_data_attribute = any([attr[0] == attribute_ for attr in data_attributes])
        if not is_attribute_a_data_attribute:
            raise ValueError(
                "all attribute of '{}' should be one of the data attribute. "
                " Got '{}' while candidates are {}.".format(
                    parameter_name, attribute_, [attr[0] for attr in data_attributes]
                )
            )


def check_datasets_active(
    dataset_ids: List[int], raise_error_if_not_exist: bool = True,
) -> Dict[int, bool]:
    """
    Check if the dataset ids provided are active.

    Raises an error if a dataset_id in the given list
    of dataset_ids does not exist on the server.

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
    dataset_list = list_datasets(status="all", data_id=dataset_ids)
    active = {}

    for did in dataset_ids:
        dataset = dataset_list.get(did, None)
        if dataset is None:
            if raise_error_if_not_exist:
                raise ValueError(f"Could not find dataset {did} in OpenML dataset list.")
        else:
            active[did] = dataset["status"] == "active"

    return active


def _name_to_id(
    dataset_name: str, version: Optional[int] = None, error_if_multiple: bool = False
) -> int:
    """ Attempt to find the dataset id of the dataset with the given name.

    If multiple datasets with the name exist, and ``error_if_multiple`` is ``False``,
    then return the least recent still active dataset.

    Raises an error if no dataset with the name is found.
    Raises an error if a version is specified but it could not be found.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset for which to find its id.
    version : int
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
    candidates = list_datasets(data_name=dataset_name, status=status, data_version=version)
    if error_if_multiple and len(candidates) > 1:
        raise ValueError("Multiple active datasets exist with name {}".format(dataset_name))
    if len(candidates) == 0:
        no_dataset_for_name = "No active datasets exist with name {}".format(dataset_name)
        and_version = " and version {}".format(version) if version is not None else ""
        raise RuntimeError(no_dataset_for_name + and_version)

    # Dataset ids are chronological so we can just sort based on ids (instead of version)
    return sorted(candidates)[0]


def get_datasets(
    dataset_ids: List[Union[str, int]], download_data: bool = True, download_qualities: bool = True
) -> List[OpenMLDataset]:
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
            get_dataset(dataset_id, download_data, download_qualities=download_qualities)
        )
    return datasets


@openml.utils.thread_safe_if_oslo_installed
def get_dataset(
    dataset_id: Union[int, str],
    download_data: bool = True,
    version: int = None,
    error_if_multiple: bool = False,
    cache_format: str = "pickle",
    download_qualities: bool = True,
) -> OpenMLDataset:
    """ Download the OpenML dataset representation, optionally also download actual data file.

    This function is thread/multiprocessing safe.
    This function uses caching. A check will be performed to determine if the information has
    previously been downloaded, and if so be loaded from disk instead of retrieved from the server.

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
    cache_format : str (default='pickle')
        Format for caching the dataset - may be feather or pickle
        Note that the default 'pickle' option may load slower than feather when
        no.of.rows is very high.
    download_qualities : bool (default=True)
        Option to download 'qualities' meta-data in addition to the minimal dataset description.
    Returns
    -------
    dataset : :class:`openml.OpenMLDataset`
        The downloaded dataset.
    """
    if cache_format not in ["feather", "pickle"]:
        raise ValueError(
            "cache_format must be one of 'feather' or 'pickle. "
            "Invalid format specified: {}".format(cache_format)
        )

    if isinstance(dataset_id, str):
        try:
            dataset_id = int(dataset_id)
        except ValueError:
            dataset_id = _name_to_id(dataset_id, version, error_if_multiple)  # type: ignore
    elif not isinstance(dataset_id, int):
        raise TypeError(
            "`dataset_id` must be one of `str` or `int`, not {}.".format(type(dataset_id))
        )

    did_cache_dir = _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, dataset_id,)

    remove_dataset_cache = True
    try:
        description = _get_dataset_description(did_cache_dir, dataset_id)
        features_file = _get_dataset_features_file(did_cache_dir, dataset_id)

        try:
            if download_qualities:
                qualities_file = _get_dataset_qualities_file(did_cache_dir, dataset_id)
            else:
                qualities_file = ""
        except OpenMLServerException as e:
            if e.code == 362 and str(e) == "No qualities found - None":
                logger.warning("No qualities found for dataset {}".format(dataset_id))
                qualities_file = None
            else:
                raise

        arff_file = _get_dataset_arff(description) if download_data else None
        if "oml:minio_url" in description and download_data:
            try:
                parquet_file = _get_dataset_parquet(description)
            except urllib3.exceptions.MaxRetryError:
                parquet_file = None
        else:
            parquet_file = None
        remove_dataset_cache = False
    except OpenMLServerException as e:
        # if there was an exception,
        # check if the user had access to the dataset
        if e.code == 112:
            raise OpenMLPrivateDatasetError(e.message) from None
        else:
            raise e
    finally:
        if remove_dataset_cache:
            _remove_cache_dir_for_id(DATASETS_CACHE_DIR_NAME, did_cache_dir)

    dataset = _create_dataset_from_description(
        description, features_file, qualities_file, arff_file, parquet_file, cache_format
    )
    return dataset


def attributes_arff_from_df(df):
    """ Describe attributes of the dataframe according to ARFF specification.

    Parameters
    ----------
    df : DataFrame, shape (n_samples, n_features)
        The dataframe containing the data set.

    Returns
    -------
    attributes_arff : str
        The data set attributes as required by the ARFF format.
    """
    PD_DTYPES_TO_ARFF_DTYPE = {"integer": "INTEGER", "floating": "REAL", "string": "STRING"}
    attributes_arff = []

    if not all([isinstance(column_name, str) for column_name in df.columns]):
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
                    "The column '{}' of the dataframe is of "
                    "'category' dtype. Therefore, all values in "
                    "this columns should be string. Please "
                    "convert the entries which are not string. "
                    "Got {} dtype in this column.".format(column_name, categories_dtype)
                )
            attributes_arff.append((column_name, categories.tolist()))
        elif column_dtype == "boolean":
            # boolean are encoded as categorical.
            attributes_arff.append((column_name, ["True", "False"]))
        elif column_dtype in PD_DTYPES_TO_ARFF_DTYPE.keys():
            attributes_arff.append((column_name, PD_DTYPES_TO_ARFF_DTYPE[column_dtype]))
        else:
            raise ValueError(
                "The dtype '{}' of the column '{}' is not "
                "currently supported by liac-arff. Supported "
                "dtypes are categorical, string, integer, "
                "floating, and boolean.".format(column_dtype, column_name)
            )
    return attributes_arff


def create_dataset(
    name,
    description,
    creator,
    contributor,
    collection_date,
    language,
    licence,
    attributes,
    data,
    default_target_attribute,
    ignore_attribute,
    citation,
    row_id_attribute=None,
    original_data_url=None,
    paper_url=None,
    update_comment=None,
    version_label=None,
):
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
        Dataset description."""

    if isinstance(data, pd.DataFrame):
        # infer the row id from the index of the dataset
        if row_id_attribute is None:
            row_id_attribute = data.index.name
        # When calling data.values, the index will be skipped.
        # We need to reset the index such that it is part of the data.
        if data.index.name is not None:
            data = data.reset_index()

    if attributes == "auto" or isinstance(attributes, dict):
        if not hasattr(data, "columns"):
            raise ValueError(
                "Automatically inferring attributes requires "
                "a pandas DataFrame. A {!r} was given instead.".format(data)
            )
        # infer the type of data for each column of the DataFrame
        attributes_ = attributes_arff_from_df(data)
        if isinstance(attributes, dict):
            # override the attributes which was specified by the user
            for attr_idx in range(len(attributes_)):
                attr_name = attributes_[attr_idx][0]
                if attr_name in attributes.keys():
                    attributes_[attr_idx] = (attr_name, attributes[attr_name])
    else:
        attributes_ = attributes
    ignore_attributes = _expand_parameter(ignore_attribute)
    _validated_data_attributes(ignore_attributes, attributes_, "ignore_attribute")

    default_target_attributes = _expand_parameter(default_target_attribute)
    _validated_data_attributes(default_target_attributes, attributes_, "default_target_attribute")

    if row_id_attribute is not None:
        is_row_id_an_attribute = any([attr[0] == row_id_attribute for attr in attributes_])
        if not is_row_id_an_attribute:
            raise ValueError(
                "'row_id_attribute' should be one of the data attribute. "
                " Got '{}' while candidates are {}.".format(
                    row_id_attribute, [attr[0] for attr in attributes_]
                )
            )

    if hasattr(data, "columns"):
        if all(isinstance(dtype, pd.SparseDtype) for dtype in data.dtypes):
            data = data.sparse.to_coo()
            # liac-arff only support COO matrices with sorted rows
            row_idx_sorted = np.argsort(data.row)
            data.row = data.row[row_idx_sorted]
            data.col = data.col[row_idx_sorted]
            data.data = data.data[row_idx_sorted]
        else:
            data = data.values

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
                "data. Got {!r} instead.".format(data[0])
            )
    elif isinstance(data, coo_matrix):
        data_format = "sparse_arff"
    else:
        raise ValueError(
            "When giving a list or a numpy.ndarray, "
            "they should contain a list/ numpy.ndarray "
            "for dense data or a dictionary for sparse "
            "data. Got {!r} instead.".format(data[0])
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
    except arff.ArffException:
        raise ValueError(
            "The arguments you have provided \
                             do not construct a valid ARFF file"
        )

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


def status_update(data_id, status):
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
        raise ValueError("Illegal status value. " "Legal values: %s" % legal_status)
    data = {"data_id": data_id, "status": status}
    result_xml = openml._api_calls._perform_api_call("data/status/update", "post", data=data)
    result = xmltodict.parse(result_xml)
    server_data_id = result["oml:data_status_update"]["oml:id"]
    server_status = result["oml:data_status_update"]["oml:status"]
    if status != server_status or int(data_id) != int(server_data_id):
        # This should never happen
        raise ValueError("Data id/status does not collide")


def edit_dataset(
    data_id,
    description=None,
    creator=None,
    contributor=None,
    collection_date=None,
    language=None,
    default_target_attribute=None,
    ignore_attribute=None,
    citation=None,
    row_id_attribute=None,
    original_data_url=None,
    paper_url=None,
) -> int:
    """ Edits an OpenMLDataset.

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
        raise TypeError("`data_id` must be of type `int`, not {}.".format(type(data_id)))

    # compose data edit parameters as xml
    form_data = {"data_id": data_id}
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

    file_elements = {"edit_parameters": ("description.xml", xmltodict.unparse(xml))}
    result_xml = openml._api_calls._perform_api_call(
        "data/edit", "post", data=form_data, file_elements=file_elements
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
        raise TypeError("`data_id` must be of type `int`, not {}.".format(type(data_id)))
    # compose data fork parameters
    form_data = {"data_id": data_id}
    result_xml = openml._api_calls._perform_api_call("data/fork", "post", data=form_data)
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_fork"]["oml:id"]
    return int(data_id)


def _topic_add_dataset(data_id: int, topic: str):
    """
    Adds a topic for a dataset.
    This API is not available for all OpenML users and is accessible only by admins.
    Parameters
    ----------
    data_id : int
        id of the dataset for which the topic needs to be added
    topic : str
        Topic to be added for the dataset
   """
    if not isinstance(data_id, int):
        raise TypeError("`data_id` must be of type `int`, not {}.".format(type(data_id)))
    form_data = {"data_id": data_id, "topic": topic}
    result_xml = openml._api_calls._perform_api_call("data/topicadd", "post", data=form_data)
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_topic"]["oml:id"]
    return int(data_id)


def _topic_delete_dataset(data_id: int, topic: str):
    """
    Removes a topic from a dataset.
    This API is not available for all OpenML users and is accessible only by admins.
    Parameters
    ----------
    data_id : int
        id of the dataset to be forked
    topic : str
        Topic to be deleted

   """
    if not isinstance(data_id, int):
        raise TypeError("`data_id` must be of type `int`, not {}.".format(type(data_id)))
    form_data = {"data_id": data_id, "topic": topic}
    result_xml = openml._api_calls._perform_api_call("data/topicdelete", "post", data=form_data)
    result = xmltodict.parse(result_xml)
    data_id = result["oml:data_topic"]["oml:id"]
    return int(data_id)


def _get_dataset_description(did_cache_dir, dataset_id):
    """Get the dataset description as xml dictionary.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str
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
    description_file = os.path.join(did_cache_dir, "description.xml")

    try:
        with io.open(description_file, encoding="utf8") as fh:
            dataset_xml = fh.read()
    except Exception:
        url_extension = "data/{}".format(dataset_id)
        dataset_xml = openml._api_calls._perform_api_call(url_extension, "get")
        with io.open(description_file, "w", encoding="utf8") as fh:
            fh.write(dataset_xml)

    description = xmltodict.parse(dataset_xml)["oml:data_set_description"]

    return description


def _get_dataset_parquet(
    description: Union[Dict, OpenMLDataset], cache_directory: str = None
) -> Optional[str]:
    """ Return the path to the local parquet file of the dataset. If is not cached, it is downloaded.

    Checks if the file is in the cache, if yes, return the path to the file.
    If not, downloads the file and caches it, then returns the file path.
    The cache directory is generated based on dataset information, but can also be specified.

    This function is NOT thread/multiprocessing safe.
    Unlike the ARFF equivalent, checksums are not available/used (for now).

    Parameters
    ----------
    description : dictionary or OpenMLDataset
        Either a dataset description as dict or OpenMLDataset.

    cache_directory: str, optional (default=None)
        Folder to store the parquet file in.
        If None, use the default cache directory for the dataset.

    Returns
    -------
    output_filename : string, optional
        Location of the Parquet file if successfully downloaded, None otherwise.
    """
    if isinstance(description, dict):
        url = description.get("oml:minio_url")
        did = description.get("oml:id")
    elif isinstance(description, OpenMLDataset):
        url = description._minio_url
        did = description.dataset_id
    else:
        raise TypeError("`description` should be either OpenMLDataset or Dict.")

    if cache_directory is None:
        cache_directory = _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, did)
    output_file_path = os.path.join(cache_directory, "dataset.pq")

    if not os.path.isfile(output_file_path):
        try:
            openml._api_calls._download_minio_file(
                source=cast(str, url), destination=output_file_path
            )
        except FileNotFoundError:
            return None
    return output_file_path


def _get_dataset_arff(description: Union[Dict, OpenMLDataset], cache_directory: str = None) -> str:
    """ Return the path to the local arff file of the dataset. If is not cached, it is downloaded.

    Checks if the file is in the cache, if yes, return the path to the file.
    If not, downloads the file and caches it, then returns the file path.
    The cache directory is generated based on dataset information, but can also be specified.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    description : dictionary or OpenMLDataset
        Either a dataset description as dict or OpenMLDataset.

    cache_directory: str, optional (default=None)
        Folder to store the arff file in.
        If None, use the default cache directory for the dataset.

    Returns
    -------
    output_filename : string
        Location of ARFF file.
    """
    if isinstance(description, dict):
        md5_checksum_fixture = description.get("oml:md5_checksum")
        url = description["oml:url"]
        did = description.get("oml:id")
    elif isinstance(description, OpenMLDataset):
        md5_checksum_fixture = description.md5_checksum
        url = description.url
        did = description.dataset_id
    else:
        raise TypeError("`description` should be either OpenMLDataset or Dict.")

    if cache_directory is None:
        cache_directory = _create_cache_directory_for_id(DATASETS_CACHE_DIR_NAME, did)
    output_file_path = os.path.join(cache_directory, "dataset.arff")

    try:
        openml._api_calls._download_text_file(
            source=url, output_path=output_file_path, md5_checksum=md5_checksum_fixture
        )
    except OpenMLHashException as e:
        additional_info = " Raised when downloading dataset {}.".format(did)
        e.args = (e.args[0] + additional_info,)
        raise

    return output_file_path


def _get_dataset_features_file(did_cache_dir: str, dataset_id: int) -> str:
    """API call to load dataset features. Loads from cache or downloads them.

    Features are feature descriptions for each column.
    (name, index, categorical, ...)

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset

    dataset_id : int
        Dataset ID

    Returns
    -------
    str
        Path of the cached dataset feature file
    """
    features_file = os.path.join(did_cache_dir, "features.xml")

    # Dataset features aren't subject to change...
    if not os.path.isfile(features_file):
        url_extension = "data/features/{}".format(dataset_id)
        features_xml = openml._api_calls._perform_api_call(url_extension, "get")
        with io.open(features_file, "w", encoding="utf8") as fh:
            fh.write(features_xml)

    return features_file


def _get_dataset_qualities_file(did_cache_dir, dataset_id):
    """API call to load dataset qualities. Loads from cache or downloads them.

    Features are metafeatures (number of features, number of classes, ...)

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset

    dataset_id : int
        Dataset ID

    download_qualities : bool
        wheather to download/use cahsed version or not.
    Returns
    -------
    str
        Path of the cached qualities file
    """
    # Dataset qualities are subject to change and must be fetched every time
    qualities_file = os.path.join(did_cache_dir, "qualities.xml")
    try:
        with io.open(qualities_file, encoding="utf8") as fh:
            qualities_xml = fh.read()
    except (OSError, IOError):
        url_extension = "data/qualities/{}".format(dataset_id)
        qualities_xml = openml._api_calls._perform_api_call(url_extension, "get")
        with io.open(qualities_file, "w", encoding="utf8") as fh:
            fh.write(qualities_xml)
    return qualities_file


def _create_dataset_from_description(
    description: Dict[str, str],
    features_file: str,
    qualities_file: str,
    arff_file: str = None,
    parquet_file: str = None,
    cache_format: str = "pickle",
) -> OpenMLDataset:
    """Create a dataset object from a description dict.

    Parameters
    ----------
    description : dict
        Description of a dataset in xml dict.
    featuresfile : str
        Path of the dataset features as xml file.
    qualities : list
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
        data_format=description["oml:format"],
        dataset_id=description["oml:id"],
        version=description["oml:version"],
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
        data_file=arff_file,
        cache_format=cache_format,
        features_file=features_file,
        qualities_file=qualities_file,
        minio_url=description.get("oml:minio_url"),
        parquet_file=parquet_file,
    )


def _get_online_dataset_arff(dataset_id):
    """Download the ARFF file for a given dataset id
    from the OpenML website.

    Parameters
    ----------
    dataset_id : int
        A dataset id.

    Returns
    -------
    str
        A string representation of an ARFF file.
    """
    dataset_xml = openml._api_calls._perform_api_call("data/%d" % dataset_id, "get")
    # build a dict from the xml.
    # use the url from the dataset description and return the ARFF string
    return openml._api_calls._download_text_file(
        xmltodict.parse(dataset_xml)["oml:data_set_description"]["oml:url"],
    )


def _get_online_dataset_format(dataset_id):
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
    return xmltodict.parse(dataset_xml)["oml:data_set_description"]["oml:format"].lower()
