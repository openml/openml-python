from collections import OrderedDict
import hashlib
import io
import os
import re
import shutil

from oslo_concurrency import lockutils
import xmltodict

from .dataset import OpenMLDataset
from ..exceptions import OpenMLCacheException, OpenMLServerNoResult
from .. import config
from .._api_calls import _perform_api_call, _read_url


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

    dataset_cache = config.get_cache_directory()
    dataset_cache_dir = os.path.join(dataset_cache, "datasets")
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
    cache_dir = config.get_cache_directory()
    did_cache_dir = os.path.join(cache_dir, "datasets", str(dataset_id))
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
    cache_dir = config.get_cache_directory()
    did_cache_dir = os.path.join(cache_dir, "datasets", str(dataset_id))
    features_file = os.path.join(did_cache_dir, "features.xml")
    try:
        with io.open(features_file, encoding='utf8') as fh:
            features_xml = fh.read()
            return xmltodict.parse(features_xml)["oml:data_features"]
    except (IOError, OSError):
        raise OpenMLCacheException("Dataset features for dataset id %d not "
                                   "cached" % dataset_id)


def _get_cached_dataset_qualities(dataset_id):
    cache_dir = config.get_cache_directory()
    did_cache_dir = os.path.join(cache_dir, "datasets", str(dataset_id))
    qualities_file = os.path.join(did_cache_dir, "qualities.xml")
    try:
        with io.open(qualities_file, encoding='utf8') as fh:
            qualities_xml = fh.read()
            return xmltodict.parse(qualities_xml)["oml:data_qualities"]['oml:quality']
    except (IOError, OSError):
        raise OpenMLCacheException("Dataset qualities for dataset id %d not "
                                   "cached" % dataset_id)


def _get_cached_dataset_arff(dataset_id):
    cache_dir = config.get_cache_directory()
    did_cache_dir = os.path.join(cache_dir, "datasets", str(dataset_id))
    output_file = os.path.join(did_cache_dir, "dataset.arff")

    try:
        with io.open(output_file, encoding='utf8'):
            pass
        return output_file
    except (OSError, IOError):
        raise OpenMLCacheException("ARFF file for dataset id %d not "
                                   "cached" % dataset_id)


def list_datasets(offset=None, size=None, tag=None):
    """Return a list of all dataset which are on OpenML.

    Parameters
    ----------
    offset : int, optional
        The number of datasets to skip, starting from the first.
    size : int, optional
        The maximum number of datasets to show.
    tag : str, optional
        Only include datasets matching this tag.

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
    api_call = "data/list"
    if offset is not None:
        api_call += "/offset/%d" % int(offset)

    if size is not None:
        api_call += "/limit/%d" % int(size)

    if tag is not None:
        api_call += "/tag/%s" % tag

    return _list_datasets(api_call)


def _list_datasets(api_call):
    # TODO add proper error handling here!
    try:
        xml_string = _perform_api_call(api_call)
    except OpenMLServerNoResult:
        return []
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
    dataset_id : iterable
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
    ddataset_id : int
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
        lock_path=os.path.join(config.get_cache_directory(), 'locks'),
    ):
        did_cache_dir = _create_dataset_cache_directory(dataset_id)

        try:
            description = _get_dataset_description(did_cache_dir, dataset_id)
            arff_file = _get_dataset_arff(did_cache_dir, description)
            features = _get_dataset_features(did_cache_dir, dataset_id)
            qualities = _get_dataset_qualities(did_cache_dir, dataset_id)
        except Exception as e:
            _remove_dataset_cache_dir(did_cache_dir)
            raise e

        dataset = _create_dataset_from_description(
            description, features, qualities, arff_file
        )
    return dataset


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
    except (OpenMLCacheException):
        dataset_xml = _perform_api_call("data/%d" % dataset_id)

        with io.open(description_file, "w", encoding='utf8') as fh:
            fh.write(dataset_xml)

    description = xmltodict.parse(dataset_xml)[
        "oml:data_set_description"]

    return description


def _get_dataset_arff(did_cache_dir, description):
    """Get the filepath to the dataset arff

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
        Location of arff file.
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
    arff_string = _read_url(url)
    md5 = hashlib.md5()
    md5.update(arff_string.encode('utf8'))
    md5_checksum = md5.hexdigest()
    if md5_checksum != md5_checksum_fixture:
        raise ValueError(
            'Checksum %s of downloaded dataset %d is unequal to the checksum '
            '%s sent by the server.' % (md5_checksum, did, md5_checksum_fixture)
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
        features_xml = _perform_api_call("data/features/%d" % dataset_id)

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
        qualities_xml = _perform_api_call("data/qualities/%d" % dataset_id)

        with io.open(qualities_file, "w", encoding='utf8') as fh:
            fh.write(qualities_xml)

    qualities = xmltodict.parse(qualities_xml, force_list=('oml:quality',))['oml:data_qualities']['oml:quality']

    return qualities


def _create_dataset_cache_directory(dataset_id):
    """Create a dataset cache directory

    In order to have a clearer cache structure and because every dataset
    is cached in several files (description, arff, features, qualities), there
    is a directory for each dataset witch the dataset ID being the directory
    name. This function creates this cache directory.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    did : int
        Dataset ID

    Returns
    -------
    str
        Path of the created dataset cache directory.
    """
    dataset_cache_dir = os.path.join(
        config.get_cache_directory(),
        "datasets",
        str(dataset_id),
    )
    if os.path.exists(dataset_cache_dir) and os.path.isdir(dataset_cache_dir):
        pass
    elif os.path.exists(dataset_cache_dir) and not os.path.isdir(dataset_cache_dir):
        raise ValueError('Dataset cache dir exists but is not a directory!')
    else:
        os.makedirs(dataset_cache_dir)
    return dataset_cache_dir


def _remove_dataset_cache_dir(did_cache_dir):
    """Remove the dataset cache directory

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    """
    try:
        shutil.rmtree(did_cache_dir)
    except (OSError, IOError):
        raise ValueError('Cannot remove faulty dataset cache directory %s.'
                         'Please do this manually!' % did_cache_dir)


def _create_dataset_from_description(description, features, qualities, arff_file):
    """Create a dataset object from a description dict.

    Parameters
    ----------
    description : dict
        Description of a dataset in xmlish dict.
    arff_file : string
        Path of dataset arff file.

    Returns
    -------
    dataset : dataset object
        Dataset object from dict and arff.
    """
    dataset = OpenMLDataset(
        description["oml:id"],
        description["oml:name"],
        description["oml:version"],
        description.get("oml:description"),
        description["oml:format"],
        description.get("oml:creator"),
        description.get("oml:contributor"),
        description.get("oml:collection_date"),
        description.get("oml:upload_date"),
        description.get("oml:language"),
        description.get("oml:licence"),
        description["oml:url"],
        description.get("oml:default_target_attribute"),
        description.get("oml:row_id_attribute"),
        description.get("oml:ignore_attribute"),
        description.get("oml:version_label"),
        description.get("oml:citation"),
        description.get("oml:tag"),
        description.get("oml:visibility"),
        description.get("oml:original_data_url"),
        description.get("oml:paper_url"),
        description.get("oml:update_comment"),
        description.get("oml:md5_checksum"),
        data_file=arff_file,
        features=features,
        qualities=qualities)
    return dataset
