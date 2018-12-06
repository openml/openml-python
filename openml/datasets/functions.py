import hashlib
import io
import os
import re

import numpy as np
import six
import arff
import pandas as pd

import xmltodict
from scipy.sparse import coo_matrix
from oslo_concurrency import lockutils
from collections import OrderedDict
from warnings import warn

import openml.utils
import openml._api_calls
from .dataset import OpenMLDataset
from ..exceptions import (
    OpenMLCacheException,
    OpenMLHashException,
    OpenMLServerException,
    PrivateDatasetError,
)
from ..utils import (
    _create_cache_directory,
    _remove_cache_dir_for_id,
    _create_cache_directory_for_id,
    _create_lockfiles_dir,
)


DATASETS_CACHE_DIR_NAME = 'datasets'



############################################################################
# Local getters/accessors to the cache directory


def _list_cached_datasets():
    """Return list with ids of all cached datasets

    Returns
    -------
    list
        List with IDs of all cached datasets.
    """
    datasets = []

    dataset_cache_dir = _create_cache_directory(DATASETS_CACHE_DIR_NAME)
    directory_content = os.listdir(dataset_cache_dir)
    directory_content.sort()

    # Find all dataset ids for which we have downloaded the dataset
    # description
    for directory_name in directory_content:
        # First check if the directory name could be an OpenML dataset id
        if not re.match(r"[0-9]*", directory_name):
            continue

        dataset_id = int(directory_name)

        directory_name = os.path.join(dataset_cache_dir,
                                      directory_name)
        dataset_directory_content = os.listdir(directory_name)

        if ("dataset.arff" in dataset_directory_content and
                "description.xml" in dataset_directory_content):
            if dataset_id not in datasets:
                datasets.append(dataset_id)

    datasets.sort()
    return datasets


def _get_cached_datasets():
    """Searches for all OpenML datasets in the OpenML cache dir.

    Return a dictionary which maps dataset ids to dataset objects"""
    dataset_list = _list_cached_datasets()
    datasets = OrderedDict()

    for dataset_id in dataset_list:
        datasets[dataset_id] = _get_cached_dataset(dataset_id)

    return datasets


def _get_cached_dataset(dataset_id):
    """Get cached dataset for ID.

    Returns
    -------
    OpenMLDataset
    """
    description = _get_cached_dataset_description(dataset_id)
    arff_file = _get_cached_dataset_arff(dataset_id)
    features = _get_cached_dataset_features(dataset_id)
    qualities = _get_cached_dataset_qualities(dataset_id)
    dataset = _create_dataset_from_description(description, features, qualities, arff_file)

    return dataset


def _get_cached_dataset_description(dataset_id):
    did_cache_dir = _create_cache_directory_for_id(
        DATASETS_CACHE_DIR_NAME, dataset_id,
    )
    description_file = os.path.join(did_cache_dir, "description.xml")
    try:
        with io.open(description_file, encoding='utf8') as fh:
            dataset_xml = fh.read()
        return xmltodict.parse(dataset_xml)["oml:data_set_description"]
    except (IOError, OSError):
        raise OpenMLCacheException(
            "Dataset description for dataset id %d not "
            "cached" % dataset_id)


def _get_cached_dataset_features(dataset_id):
    did_cache_dir = _create_cache_directory_for_id(
        DATASETS_CACHE_DIR_NAME, dataset_id,
    )
    features_file = os.path.join(did_cache_dir, "features.xml")
    try:
        with io.open(features_file, encoding='utf8') as fh:
            features_xml = fh.read()
            return xmltodict.parse(features_xml)["oml:data_features"]
    except (IOError, OSError):
        raise OpenMLCacheException("Dataset features for dataset id %d not "
                                   "cached" % dataset_id)


def _get_cached_dataset_qualities(dataset_id):
    did_cache_dir = _create_cache_directory_for_id(
        DATASETS_CACHE_DIR_NAME, dataset_id,
    )
    qualities_file = os.path.join(did_cache_dir, "qualities.xml")
    try:
        with io.open(qualities_file, encoding='utf8') as fh:
            qualities_xml = fh.read()
            return xmltodict.parse(qualities_xml)["oml:data_qualities"]['oml:quality']
    except (IOError, OSError):
        raise OpenMLCacheException("Dataset qualities for dataset id %d not "
                                   "cached" % dataset_id)


def _get_cached_dataset_arff(dataset_id):
    did_cache_dir = _create_cache_directory_for_id(
        DATASETS_CACHE_DIR_NAME, dataset_id,
    )
    output_file = os.path.join(did_cache_dir, "dataset.arff")

    try:
        with io.open(output_file, encoding='utf8'):
            pass
        return output_file
    except (OSError, IOError):
        raise OpenMLCacheException("ARFF file for dataset id %d not "
                                   "cached" % dataset_id)


def list_datasets(offset=None, size=None, status=None, tag=None, **kwargs):

    """
    Return a list of all dataset which are on OpenML. (Supports large amount of results)

    Parameters
    ----------
    offset : int, optional
        The number of datasets to skip, starting from the first.
    size : int, optional
        The maximum number of datasets to show.
    status : str, optional
        Should be {active, in_preparation, deactivated}. By
        default active datasets are returned, but also datasets
        from another status can be requested.
    tag : str, optional
    kwargs : dict, optional
        Legal filter operators (keys in the dict):
        data_name, data_version, number_instances,
        number_features, number_classes, number_missing_values.

    Returns
    -------
    datasets : dict of dicts
        A mapping from dataset ID to dict.

        Every dataset is represented by a dictionary containing
        the following information:
        - dataset id
        - name
        - format
        - status

        If qualities are calculated for the dataset, some of
        these are also returned.
    """

    return openml.utils._list_all(_list_datasets, offset=offset, size=size, status=status, tag=tag, **kwargs)


def _list_datasets(**kwargs):

    """
    Perform api call to return a list of all datasets.

    Parameters
    ----------
    kwargs : dict, optional
        Legal filter operators (keys in the dict):
        {tag, status, limit, offset, data_name, data_version, number_instances,
        number_features, number_classes, number_missing_values.

    Returns
    -------
    datasets : dict of dicts
    """

    api_call = "data/list"

    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)
    return __list_datasets(api_call)


def __list_datasets(api_call):

    xml_string = openml._api_calls._perform_api_call(api_call)
    datasets_dict = xmltodict.parse(xml_string, force_list=('oml:dataset',))

    # Minimalistic check if the XML is useful
    assert type(datasets_dict['oml:data']['oml:dataset']) == list, \
        type(datasets_dict['oml:data'])
    assert datasets_dict['oml:data']['@xmlns:oml'] == \
        'http://openml.org/openml', datasets_dict['oml:data']['@xmlns:oml']

    datasets = dict()
    for dataset_ in datasets_dict['oml:data']['oml:dataset']:
        did = int(dataset_['oml:did'])
        dataset = {'did': did,
                   'name': dataset_['oml:name'],
                   'format': dataset_['oml:format'],
                   'status': dataset_['oml:status']}

        # The number of qualities can range from 0 to infinity
        for quality in dataset_.get('oml:quality', list()):
            quality['#text'] = float(quality['#text'])
            if abs(int(quality['#text']) - quality['#text']) < 0.0000001:
                quality['#text'] = int(quality['#text'])
            dataset[quality['@name']] = quality['#text']
        datasets[did] = dataset

    return datasets


def check_datasets_active(dataset_ids):
    """Check if the dataset ids provided are active.

    Parameters
    ----------
    dataset_ids : iterable
        Integers representing dataset ids.

    Returns
    -------
    dict
        A dictionary with items {did: bool}
    """
    dataset_list = list_datasets()
    dataset_ids = sorted(dataset_ids)
    active = {}

    for dataset in dataset_list:
        active[dataset['did']] = dataset['status'] == 'active'

    for did in dataset_ids:
        if did not in active:
            raise ValueError('Could not find dataset %d in OpenML dataset list.'
                             % did)

    active = {did: active[did] for did in dataset_ids}

    return active


def get_datasets(dataset_ids):
    """Download datasets.

    This function iterates :meth:`openml.datasets.get_dataset`.

    Parameters
    ----------
    dataset_ids : iterable
        Integers representing dataset ids.

    Returns
    -------
    datasets : list of datasets
        A list of dataset objects.
    """
    datasets = []
    for dataset_id in dataset_ids:
        datasets.append(get_dataset(dataset_id))
    return datasets


def get_dataset(dataset_id):
    """Download a dataset.

    TODO: explain caching!

    This function is thread/multiprocessing safe.

    Parameters
    ----------
    dataset_id : int
        Dataset ID of the dataset to download

    Returns
    -------
    dataset : :class:`openml.OpenMLDataset`
        The downloaded dataset."""
    try:
        dataset_id = int(dataset_id)
    except:
        raise ValueError("Dataset ID is neither an Integer nor can be "
                         "cast to an Integer.")

    with lockutils.external_lock(
        name='datasets.functions.get_dataset:%d' % dataset_id,
        lock_path=_create_lockfiles_dir(),
    ):
        did_cache_dir = _create_cache_directory_for_id(
            DATASETS_CACHE_DIR_NAME, dataset_id,
        )

        try:
            remove_dataset_cache = True
            description = _get_dataset_description(did_cache_dir, dataset_id)
            arff_file = _get_dataset_arff(did_cache_dir, description)
            features = _get_dataset_features(did_cache_dir, dataset_id)
            qualities = _get_dataset_qualities(did_cache_dir, dataset_id)
            remove_dataset_cache = False
        except OpenMLServerException as e:
            # if there was an exception, check if the user had access to the dataset
            if e.code == 112:
                six.raise_from(PrivateDatasetError(e.message), None)
            else:
                raise e
        finally:
            if remove_dataset_cache:
                _remove_cache_dir_for_id(DATASETS_CACHE_DIR_NAME, did_cache_dir)

        dataset = _create_dataset_from_description(
            description, features, qualities, arff_file
        )
    return dataset


def attributes_arff_from_df(df):
    """Create the attributes as specified by the ARFF format using a dataframe.

    Parameters
    ----------
    df : DataFrame, shape (n_samples, n_features)
        The dataframe containing the data set.

    Returns
    -------
    attributes_arff : str
        The data set attributes as required by the ARFF format.
    """
    PD_DTYPES_TO_ARFF_DTYPE = {
        'integer': 'INTEGER',
        'floating': 'REAL',
        'string': 'STRING'
    }
    attributes_arff = []
    for column_name in df:
        # skipna=True does not infer properly the dtype. The NA values are
        # dropped before the inference instead.
        column_dtype = pd.api.types.infer_dtype(df[column_name].dropna())

        if column_dtype == 'categorical':
            # for categorical feature, arff expects a list string. However, a
            # categorical column can contain mixed type and we should therefore
            # raise an error asking to convert all entries to string.
            categories = df[column_name].cat.categories
            categories_dtype = pd.api.types.infer_dtype(categories)
            if categories_dtype not in ('string', 'unicode'):
                raise ValueError("The column '{}' of the dataframe is of "
                                 "'category' dtype. Therefore, all values in "
                                 "this columns should be string. Please "
                                 "convert the entries which are not string. "
                                 "Got {} dtype in this column."
                                 .format(column_name, categories_dtype))
            attributes_arff.append((column_name, categories.tolist()))
        elif column_dtype == 'boolean':
            # boolean are encoded as categorical.
            attributes_arff.append((column_name, ['True', 'False']))
        elif column_dtype in PD_DTYPES_TO_ARFF_DTYPE.keys():
            attributes_arff.append((column_name,
                                    PD_DTYPES_TO_ARFF_DTYPE[column_dtype]))
        else:
            raise ValueError("The dtype '{}' of the column '{}' is not "
                             "currently supported by liac-arff. Supported "
                             "dtypes are categorical, string, integer, "
                             "floating, and boolean."
                             .format(column_dtype, column_name))
    return attributes_arff


def create_dataset(name, description, creator, contributor,
                   collection_date, language,
                   licence, attributes, data,
                   default_target_attribute,
                   ignore_attribute, citation,
                   row_id_attribute=None,
                   original_data_url=None, paper_url=None,
                   update_comment=None, version_label=None):
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
        Attributes that should be excluded in modelling, such as identifiers and indexes.
    citation : str
        Reference(s) that should be cited when building on this data.
    version_label : str, optional
        Version label provided by user, can be a date, hash, or some other type of id.
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

    if isinstance(data, (pd.DataFrame, pd.SparseDataFrame)):
        # infer the row id from the index of the dataset
        if row_id_attribute is None:
            row_id_attribute = data.index.name
        # When calling data.values, the index will be skipped. We need to reset
        # the index such that it is part of the data.
        if data.index.name is not None:
            data = data.reset_index()

    if attributes == 'auto' or isinstance(attributes, dict):
        if not hasattr(data, "columns"):
            raise ValueError("Automatically inferring the attributes required "
                             "a pandas DataFrame or SparseDataFrame. "
                             "A {!r} was given instead.".format(data))
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

    if row_id_attribute is not None:
        is_row_id_an_attribute = any([attr[0] == row_id_attribute
                                      for attr in attributes_])
        if not is_row_id_an_attribute:
            raise ValueError(
                "'row_id_attribute' should be one of the data attribute. "
                " Got '{}' while candidates are {}."
                .format(row_id_attribute, [attr[0] for attr in attributes_])
            )

    if hasattr(data, "columns"):
        if isinstance(data, pd.SparseDataFrame):
            data = data.to_coo()
            # liac-arff only support COO matrices with sorted rows
            row_idx_sorted = np.argsort(data.row)
            data.row = data.row[row_idx_sorted]
            data.col = data.col[row_idx_sorted]
            data.data = data.data[row_idx_sorted]
        else:
            data = data.values

    if isinstance(data, (list, np.ndarray)):
        if isinstance(data[0], (list, np.ndarray)):
            data_format = 'arff'
        elif isinstance(data[0], dict):
            data_format = 'sparse_arff'
        else:
            raise ValueError(
                'When giving a list or a numpy.ndarray, '
                'they should contain a list/ numpy.ndarray '
                'for dense data or a dictionary for sparse '
                'data. Got {!r} instead.'
                .format(data[0])
            )
    elif isinstance(data, coo_matrix):
        data_format = 'sparse_arff'
    else:
        raise ValueError(
            'When giving a list or a numpy.ndarray, '
            'they should contain a list/ numpy.ndarray '
            'for dense data or a dictionary for sparse '
            'data. Got {!r} instead.'
            .format(data[0])
        )

    arff_object = {
        'relation': name,
        'description': description,
        'attributes': attributes_,
        'data': data
    }

    # serializes the ARFF dataset object and returns a string
    arff_dataset = arff.dumps(arff_object)
    try:
        # check if ARFF is valid
        decoder = arff.ArffDecoder()
        return_type = arff.COO if data_format == 'sparse_arff' else arff.DENSE
        decoder.decode(
            arff_dataset,
            encode_nominal=True,
            return_type=return_type
        )
    except arff.ArffException:
        raise ValueError("The arguments you have provided \
                             do not construct a valid ARFF file")

    return OpenMLDataset(
        name,
        description,
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
    Updates the status of a dataset to either 'active' or 'deactivated'. Please
    see the OpenML API documentation for a description of the status and all
    legal status transitions:
    https://docs.openml.org/#dataset-status

    Parameters
    ----------
    data_id : int
        The data id of the dataset
    status : str,
        'active' or 'deactivated'
    """
    legal_status = {'active', 'deactivated'}
    if status not in legal_status:
        raise ValueError('Illegal status value. '
                         'Legal values: %s' % legal_status)
    data = {'data_id': data_id, 'status': status}
    result_xml = openml._api_calls._perform_api_call("data/status/update",
                                                     data=data)
    result = xmltodict.parse(result_xml)
    server_data_id = result['oml:data_status_update']['oml:id']
    server_status = result['oml:data_status_update']['oml:status']
    if status != server_status or int(data_id) != int(server_data_id):
        # This should never happen
        raise ValueError('Data id/status does not collide')


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

    # TODO implement a cache for this that invalidates itself after some
    # time
    # This can be saved on disk, but cannot be cached properly, because
    # it contains the information on whether a dataset is active.
    description_file = os.path.join(did_cache_dir, "description.xml")

    try:
        return _get_cached_dataset_description(dataset_id)
    except OpenMLCacheException:
        dataset_xml = openml._api_calls._perform_api_call("data/%d" % dataset_id)
        with io.open(description_file, "w", encoding='utf8') as fh:
            fh.write(dataset_xml)

    description = xmltodict.parse(dataset_xml)[
        "oml:data_set_description"]

    return description


def _get_dataset_arff(did_cache_dir, description):
    """Get the filepath to the dataset ARFF

    Checks if the file is in the cache, if yes, return the path to the file. If
    not, downloads the file and caches it, then returns the file path.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset.

    description : dictionary
        Dataset description dict.

    Returns
    -------
    output_filename : string
        Location of ARFF file.
    """
    output_file_path = os.path.join(did_cache_dir, "dataset.arff")
    md5_checksum_fixture = description.get("oml:md5_checksum")
    did = description.get("oml:id")

    # This means the file is still there; whether it is useful is up to
    # the user and not checked by the program.
    try:
        with io.open(output_file_path, encoding='utf8'):
            pass
        return output_file_path
    except (OSError, IOError):
        pass

    url = description['oml:url']
    arff_string = openml._api_calls._read_url(url)
    md5 = hashlib.md5()
    md5.update(arff_string.encode('utf-8'))
    md5_checksum = md5.hexdigest()
    if md5_checksum != md5_checksum_fixture:
        raise OpenMLHashException(
            'Checksum %s of downloaded dataset %d is unequal to the checksum '
            '%s sent by the server.' % (
                md5_checksum, int(did), md5_checksum_fixture
            )
        )

    with io.open(output_file_path, "w", encoding='utf8') as fh:
        fh.write(arff_string)
    del arff_string

    return output_file_path


def _get_dataset_features(did_cache_dir, dataset_id):
    """API call to get dataset features (cached)

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
    features : dict
        Dictionary containing dataset feature descriptions, parsed from XML.
    """
    features_file = os.path.join(did_cache_dir, "features.xml")

    # Dataset features aren't subject to change...
    try:
        with io.open(features_file, encoding='utf8') as fh:
            features_xml = fh.read()
    except (OSError, IOError):
        features_xml = openml._api_calls._perform_api_call("data/features/%d" % dataset_id)

        with io.open(features_file, "w", encoding='utf8') as fh:
            fh.write(features_xml)

    features = xmltodict.parse(features_xml, force_list=('oml:feature',))["oml:data_features"]

    return features


def _get_dataset_qualities(did_cache_dir, dataset_id):
    """API call to get dataset qualities (cached)

    Features are metafeatures (number of features, number of classes, ...)

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset

    dataset_id : int
        Dataset ID

    Returns
    -------
    qualities : dict
        Dictionary containing dataset qualities, parsed from XML.
    """
    # Dataset qualities are subject to change and must be fetched every time
    qualities_file = os.path.join(did_cache_dir, "qualities.xml")
    try:
        with io.open(qualities_file, encoding='utf8') as fh:
            qualities_xml = fh.read()
    except (OSError, IOError):
        qualities_xml = openml._api_calls._perform_api_call("data/qualities/%d" % dataset_id)

        with io.open(qualities_file, "w", encoding='utf8') as fh:
            fh.write(qualities_xml)

    qualities = xmltodict.parse(qualities_xml, force_list=('oml:quality',))['oml:data_qualities']['oml:quality']

    return qualities


def _create_dataset_from_description(description, features, qualities, arff_file):
    """Create a dataset object from a description dict.

    Parameters
    ----------
    description : dict
        Description of a dataset in xml dict.
    arff_file : string
        Path of dataset ARFF file.

    Returns
    -------
    dataset : dataset object
        Dataset object from dict and ARFF.
    """
    dataset = OpenMLDataset(
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
        default_target_attribute=description.get(
            "oml:default_target_attribute"
        ),
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
        features=features,
        qualities=qualities,
    )
    return dataset


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
    dataset_xml = openml._api_calls._perform_api_call("data/%d" % dataset_id)
    # build a dict from the xml.
    # use the url from the dataset description and return the ARFF string
    return openml._api_calls._read_url(
        xmltodict.parse(dataset_xml)['oml:data_set_description']['oml:url']
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
    dataset_xml = openml._api_calls._perform_api_call("data/%d" % dataset_id)
    # build a dict from the xml and get the format from the dataset description
    return xmltodict\
        .parse(dataset_xml)['oml:data_set_description']['oml:format']\
        .lower()
