import os
import re
import sys
from collections import OrderedDict
import xmltodict
from .dataset import OpenMLDataset
from ..exceptions import OpenMLCacheException

if sys.version_info[0] >= 3:
    from urllib.error import URLError
else:
    from urllib2 import URLError


############################################################################
# Local getters/accessors to the cache directory

def get_list_of_cached_datasets(api_connector):
    """Return list with ids of all cached datasets"""
    datasets = []

    for dataset_cache_dir in [api_connector.dataset_cache_dir,
                              api_connector._private_directory_datasets]:
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
                datasets.append(did)

    datasets.sort()
    return datasets


def get_cached_datasets(api_connector):
    """Searches for all OpenML datasets in the OpenML cache dir.

    Return a dictionary which maps dataset ids to dataset objects"""
    dataset_list = get_list_of_cached_datasets(api_connector)
    datasets = OrderedDict()

    for did in dataset_list:
        datasets[did] = get_cached_dataset(api_connector, did)

    return datasets


def get_cached_dataset(api_connector, did):
    # This code is slow...replace it with new API calls
    description = _get_cached_dataset_description(api_connector, did)
    arff_file = _get_cached_dataset_arff(api_connector, did)
    dataset = _create_dataset_from_description(description, arff_file)

    return dataset


def _get_cached_dataset_description(api_connector, did):
    for dataset_cache_dir in [api_connector.dataset_cache_dir,
                              api_connector._private_directory_datasets]:
        did_cache_dir = os.path.join(dataset_cache_dir, str(did))
        description_file = os.path.join(did_cache_dir, "description.xml")
        try:
            with open(description_file) as fh:
                dataset_xml = fh.read()
        except (IOError, OSError):
            continue

        return xmltodict.parse(dataset_xml)["oml:data_set_description"]

    raise OpenMLCacheException("Dataset description for did %d not "
                               "cached" % did)


def _get_cached_dataset_arff(api_connector, did):
    for dataset_cache_dir in [api_connector.dataset_cache_dir,
                              api_connector._private_directory_datasets]:
        did_cache_dir = os.path.join(dataset_cache_dir, str(did))
        output_file = os.path.join(did_cache_dir, "dataset.arff")

        try:
            with open(output_file):
                pass
            return output_file
        except (OSError, IOError):
            # TODO create NOTCACHEDEXCEPTION
            continue

    raise OpenMLCacheException("ARFF file for did %d not "
                               "cached" % did)


def get_dataset_list(api_connector):
    """Return a list of all dataset which are on OpenML.

    Returns
    -------
    datasets : list
        A list of all datasets. Every dataset is represented by a
        dictionary containing the following information: dataset id,
        and status. If qualities are calculated for the dataset, some of
        these are also returned.
    """
    # TODO add proper error handling here!
    return_code, xml_string = api_connector._perform_api_call("data/list/")
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


def datasets_active(api_connector, dids):
    """Check if the dataset ids provided are active.

    Parameters
    ----------
    dids : iterable
        A list of integers representing dataset ids.

    Returns
    -------
    dict
        A dictionary with items {did: active}, where active is a boolean. It
        is set to True if the dataset is active.
    """
    dataset_list = get_dataset_list(api_connector)
    dids = sorted(dids)
    active = {}

    dataset_list_idx = 0
    for did in dids:
        # TODO replace with a more efficient while loop!
        for idx in range(dataset_list_idx, len(dataset_list)):
            if did == dataset_list[idx]['did']:
                active['did'] = bool(dataset_list[idx]['status'])
        dataset_list_idx = idx


def download_datasets(api_connector, dids):
    """Download datasets.

    Parameters
    ----------
    dids : iterable
        A list of integers representing dataset ids.

    Returns
    -------
    list
        A list of dataset objects.

    Notes
    -----
    Uses :func:`download_dataset` internally. Please read
    the documentation of this.
    """
    datasets = []
    for did in dids:
        datasets.append(download_dataset(api_connector, did))
    return datasets


def download_dataset(api_connector, did):
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

    description = download_dataset_description(api_connector, did)
    arff_file = download_dataset_arff(api_connector, did, description=description)

    dataset = _create_dataset_from_description(description, arff_file)
    return dataset


def download_dataset_description(api_connector, did):
    # TODO implement a cache for this that invalidates itself after some
    # time
    # This can be saved on disk, but cannot be cached properly, because
    # it contains the information on whether a dataset is active.
    did_cache_dir = _create_dataset_cache_dir(api_connector, did)
    description_file = os.path.join(did_cache_dir, "description.xml")

    try:
        return _get_cached_dataset_description(api_connector, did)
    except (OpenMLCacheException):
        try:
            return_code, dataset_xml = api_connector._perform_api_call(
                "data/%d" % did)
        except (URLError, UnicodeEncodeError) as e:
            # TODO logger.debug
            _remove_dataset_chache_dir(api_connector, did)
            print(e)
            raise e

        with open(description_file, "w") as fh:
            fh.write(dataset_xml)

    try:
        description = xmltodict.parse(dataset_xml)[
            "oml:data_set_description"]
    except Exception as e:
        # TODO logger.debug
        _remove_dataset_chache_dir(api_connector, did)
        print("Dataset ID", did)
        raise e

    with open(description_file, "w") as fh:
        fh.write(dataset_xml)

    return description


def download_dataset_arff(api_connector, did, description=None):
    did_cache_dir = _create_dataset_cache_dir(api_connector, did)
    output_file = os.path.join(did_cache_dir, "dataset.arff")

    # This means the file is still there; whether it is useful is up to
    # the user and not checked by the program.
    try:
        with open(output_file):
            pass
        return output_file
    except (OSError, IOError):
        pass

    if description is None:
        description = download_dataset_description(api_connector, did)
    url = description['oml:url']
    return_code, arff_string = api_connector._read_url(url)
    # TODO: it is inefficient to load the dataset in memory prior to
    # saving it to the hard disk!
    with open(output_file, "w") as fh:
        fh.write(arff_string)
    del arff_string

    return output_file


def download_dataset_features(api_connector, did):
    did_cache_dir = _create_dataset_cache_dir(api_connector, did)
    features_file = os.path.join(did_cache_dir, "features.xml")

    # Dataset features aren't subject to change...
    try:
        with open(features_file) as fh:
            features_xml = fh.read()
    except (OSError, IOError):
        try:
            return_code, features_xml = api_connector._perform_api_call(
                "data/features/%d" % did)
        except (URLError, UnicodeEncodeError) as e:
            # TODO logger.debug
            print(e)
            raise e

        with open(features_file, "w") as fh:
            fh.write(features_xml)

    try:
        features = xmltodict.parse(features_xml)["oml:data_features"]
    except Exception as e:
        # TODO logger.debug
        print("Dataset ID", did)
        raise e

    return features


def download_dataset_qualities(api_connector, did):
    # Dataset qualities are subject to change and must be fetched every time
    did_cache_dir = _create_dataset_cache_dir(api_connector, did)
    qualities_file = os.path.join(did_cache_dir, "qualities.xml")
    try:
        return_code, qualities_xml = api_connector._perform_api_call(
            "data/qualities/%d" % did)
    except (URLError, UnicodeEncodeError) as e:
        # TODO logger.debug
        print(e)
        raise e

    with open(qualities_file, "w") as fh:
        fh.write(qualities_xml)

    try:
        qualities = xmltodict.parse(qualities_xml)['oml:data_qualities']
    except Exception as e:
        # TODO useful debug
        raise e

    return qualities


def _create_dataset_cache_dir(api_connector, did):
    dataset_cache_dir = os.path.join(api_connector.dataset_cache_dir, str(did))
    try:
        os.makedirs(dataset_cache_dir)
    except (OSError, IOError):
        # TODO add debug information!
        pass
    return dataset_cache_dir


def _remove_dataset_chache_dir(api_connector, did):
    dataset_cache_dir = os.path.join(api_connector.dataset_cache_dir, str(did))
    try:
        os.rmdir(dataset_cache_dir)
    except (OSError, IOError):
        # TODO add debug information
        pass


def _create_dataset_from_description(description, arff_file):
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
