import os
import re
import shutil
import sys
from collections import OrderedDict
import xmltodict
from .dataset import OpenMLDataset
from ..exceptions import OpenMLCacheException
from .. import config
from .._api_calls import _perform_api_call, _read_url

if sys.version_info[0] >= 3:
    from urllib.error import URLError
else:
    from urllib2 import URLError


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

    for dataset_cache in [config.get_cache_directory(), config.get_private_directory()]:
        dataset_cache_dir = os.path.join(dataset_cache, "datasets")
        directory_content = os.listdir(dataset_cache_dir)
        directory_content.sort()

        # Find all dataset ids for which we have downloaded the dataset
        # description
        for directory_name in directory_content:
            # First check if the directory name could be an OpenML dataset id
            if not re.match(r"[0-9]*", directory_name):
                continue

            did = int(directory_name)

            directory_name = os.path.join(dataset_cache_dir,
                                          directory_name)
            dataset_directory_content = os.listdir(directory_name)

            if "dataset.arff" in dataset_directory_content and \
                    "description.xml" in dataset_directory_content:
                if did not in datasets:
                    datasets.append(did)

    datasets.sort()
    return datasets


def _get_cached_datasets():
    """Searches for all OpenML datasets in the OpenML cache dir.

    Return a dictionary which maps dataset ids to dataset objects"""
    dataset_list = _list_cached_datasets()
    datasets = OrderedDict()

    for did in dataset_list:
        datasets[did] = _get_cached_dataset(did)

    return datasets


def _get_cached_dataset(did):
    """Get cached dataset for ID.

    Returns
    -------
    OpenMLDataset
    """
    description = _get_cached_dataset_description(did)
    arff_file = _get_cached_dataset_arff(did)
    dataset = _create_dataset_from_description(description, arff_file)

    return dataset


def _get_cached_dataset_description(did):
    for cache_dir in [config.get_cache_directory(),
                      config.get_private_directory()]:
        did_cache_dir = os.path.join(cache_dir, "datasets", str(did))
        description_file = os.path.join(did_cache_dir, "description.xml")
        try:
            with open(description_file) as fh:
                dataset_xml = fh.read()
        except (IOError, OSError):
            continue

        return xmltodict.parse(dataset_xml)["oml:data_set_description"]

    raise OpenMLCacheException("Dataset description for did %d not "
                               "cached" % did)


def _get_cached_dataset_arff(did):
    for cache_dir in [config.get_cache_directory(),
                      config.get_private_directory()]:
        did_cache_dir = os.path.join(cache_dir, "datasets", str(did))
        output_file = os.path.join(did_cache_dir, "dataset.arff")

        try:
            with open(output_file):
                pass
            return output_file
        except (OSError, IOError):
            continue

    raise OpenMLCacheException("ARFF file for did %d not "
                               "cached" % did)


def list_datasets():
    """Return a list of all dataset which are on OpenML.

    Returns
    -------
    datasets : list of dicts
        A list of all datasets. 
        
        Every dataset is represented by a dictionary containing 
        the following information: 
        - dataset id
        - status
        
        If qualities are calculated for the dataset, some of
        these are also returned.
    """
    return _list_datasets("data/list")


def list_datasets_by_tag(tag):
    """Return all datasets having the given tag.

    Returns
    -------
    datasets : list of dicts
        A list of all datasets having the given tag. Every dataset is
        represented by a dictionary containing the following information:
        dataset id, and status. If qualities are calculated for the dataset,
        some of these are also returned.

    """
    return _list_datasets("data/list/%s" % tag)


def _list_datasets(api_call):
    # TODO add proper error handling here!
    return_code, xml_string = _perform_api_call(api_call)
    datasets_dict = xmltodict.parse(xml_string)

    # Minimalistic check if the XML is useful
    assert type(datasets_dict['oml:data']['oml:dataset']) == list, \
        type(datasets_dict['oml:data'])
    assert datasets_dict['oml:data']['@xmlns:oml'] == \
        'http://openml.org/openml'

    datasets = []
    for dataset_ in datasets_dict['oml:data']['oml:dataset']:
        dataset = {'did': int(dataset_['oml:did']),
                   'name': dataset_['oml:name'],
                   'format': dataset_['oml:format'],
                   'status': dataset_['oml:status']}

        # The number of qualities can range from 0 to infinity
        for quality in dataset_.get('oml:quality', list()):
            quality['#text'] = float(quality['#text'])
            if abs(int(quality['#text']) - quality['#text']) < 0.0000001:
                quality['#text'] = int(quality['#text'])
            dataset[quality['@name']] = quality['#text']

        datasets.append(dataset)
    datasets.sort(key=lambda t: t['did'])

    return datasets


def check_datasets_active(dids):
    """Check if the dataset ids provided are active.

    Parameters
    ----------
    dids : iterable
        Integers representing dataset ids.

    Returns
    -------
    dict
        A dictionary with items {did: bool}
    """
    dataset_list = list_datasets()
    dids = sorted(dids)
    active = {}

    for dataset in dataset_list:
        active[dataset['did']] = dataset['status'] == 'active'

    for did in dids:
        if did not in active:
            raise ValueError('Could not find dataset %d in OpenML dataset list.'
                             % did)

    active = {did: active[did] for did in dids}

    return active


def get_datasets(dids):
    """Get datasets.

    This function iterates :meth:`openml.datasets.get_dataset`.

    Parameters
    ----------
    dids : iterable
        Integers representing dataset ids.

    Returns
    -------
    datasets : list of datasets
        A list of dataset objects.
    """
    datasets = []
    for did in dids:
        datasets.append(get_dataset(did))
    return datasets


def get_dataset(did):
    """Download a dataset.

    TODO: explain caching!

    Parameters
    ----------
    dids : int
        Dataset ID of the dataset to download

    Returns
    -------
    dataset : :class:`openml.OpenMLDataset`
        The downloaded dataset."""
    try:
        did = int(did)
    except:
        raise ValueError("Dataset ID is neither an Integer nor can be "
                         "cast to an Integer.")

    did_cache_dir = _create_dataset_cache_directory(did)

    try:
        description = _get_dataset_description(did_cache_dir, did)
        arff_file = _get_dataset_arff(did_cache_dir, description)
        # TODO not used yet, figure out what to do with them...
        features = _get_dataset_features(did_cache_dir, did)
        qualities = _get_dataset_qualities(did_cache_dir, did)
    except Exception as e:
        _remove_dataset_cache_dir(did_cache_dir)
        raise e

    dataset = _create_dataset_from_description(description, arff_file)
    return dataset


def _get_dataset_description(did_cache_dir, did):
    """Get the dataset description as xml dictionary

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset.

    did : int
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
        return _get_cached_dataset_description(did)
    except (OpenMLCacheException):
        return_code, dataset_xml = _perform_api_call(
            "data/%d" % did)

        with open(description_file, "w") as fh:
            fh.write(dataset_xml)

    description = xmltodict.parse(dataset_xml)[
        "oml:data_set_description"]

    with open(description_file, "w") as fh:
        fh.write(dataset_xml)

    return description


def _get_dataset_arff(did_cache_dir, description):
    """Get the filepath to the dataset arff

    Checks if the file is in the cache, if yes, return the path to the file. If
    not, downloads the file and caches it, then returns the file path.

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset.

    did : int
        Dataset ID

    description : dictionary
        Dataset description dict.

    Returns
    -------
    output_filename : string
        Location of arff file.
    """
    output_file_path = os.path.join(did_cache_dir, "dataset.arff")

    # This means the file is still there; whether it is useful is up to
    # the user and not checked by the program.
    try:
        with open(output_file_path):
            pass
        return output_file_path
    except (OSError, IOError):
        pass

    url = description['oml:url']
    return_code, arff_string = _read_url(url)

    with open(output_file_path, "w") as fh:
        fh.write(arff_string)
    del arff_string

    return output_file_path


def _get_dataset_features(did_cache_dir, did):
    """API call to get dataset features (cached)

    Features are feature descriptions for each column.
    (name, index, categorical, ...)

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset

    did : int
        Dataset ID

    Returns
    -------
    features : dict
        Dictionary containing dataset feature descriptions, parsed from XML.
    """
    features_file = os.path.join(did_cache_dir, "features.xml")

    # Dataset features aren't subject to change...
    try:
        with open(features_file) as fh:
            features_xml = fh.read()
    except (OSError, IOError):
        return_code, features_xml = _perform_api_call(
            "data/features/%d" % did)

        with open(features_file, "w") as fh:
            fh.write(features_xml)

    features = xmltodict.parse(features_xml)["oml:data_features"]

    return features


def _get_dataset_qualities(did_cache_dir, did):
    """API call to get dataset qualities (cached)

    Features are metafeatures (number of features, number of classes, ...)

    Parameters
    ----------
    did_cache_dir : str
        Cache subdirectory for this dataset

    did : int
        Dataset ID

    Returns
    -------
    qualities : dict
        Dictionary containing dataset qualities, parsed from XML.
    """
    # Dataset qualities are subject to change and must be fetched every time
    qualities_file = os.path.join(did_cache_dir, "qualities.xml")
    try:
        with open(qualities_file) as fh:
            qualities_xml = fh.read()
    except (OSError, IOError):
        return_code, qualities_xml = _perform_api_call(
            "data/qualities/%d" % did)

        with open(qualities_file, "w") as fh:
            fh.write(qualities_xml)

    qualities = xmltodict.parse(qualities_xml)['oml:data_qualities']

    return qualities


def _create_dataset_cache_directory(did):
    """Create a dataset cache directory

    In order to have a clearer cache structure and because every dataset
    is cached in several files (description, arff, features, qualities), there
    is a directory for each dataset witch the dataset ID being the directory
    name. This function creates this cache directory.

    Parameters
    ----------
    did : int
        Dataset ID

    Returns
    -------
    str
        Path of the created dataset cache directory.
    """
    dataset_cache_dir = os.path.join(config.get_cache_directory(), "datasets", str(did))
    try:
        os.makedirs(dataset_cache_dir)
    except (OSError, IOError):
        # TODO add debug information!
        pass
    return dataset_cache_dir


def _remove_dataset_cache_dir(did_cache_dir):
    """Remove the dataset cache directory

    Parameters
    ----------
    """
    try:
        os.rmdir(did_cache_dir)
    except (OSError, IOError):
        try:
            shutil.rmtree(did_cache_dir)
        except (OSError, IOError):
            raise ValueError('Cannot remove faulty dataset cache directory %s.'
                             'Please do this manually!' % did_cache_dir)


def _create_dataset_from_description(description, arff_file):
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
        data_file=arff_file)
    return dataset
