import os
import hashlib
import xmltodict
import shutil
import warnings
import pandas as pd
from functools import wraps

import openml._api_calls
import openml.exceptions
from . import config

oslo_installed = False
try:
    # Currently, importing oslo raises a lot of warning that it will stop working
    # under python3.8; remove this once they disappear
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from oslo_concurrency import lockutils
        oslo_installed = True
except ImportError:
    pass


def extract_xml_tags(xml_tag_name, node, allow_none=True):
    """Helper to extract xml tags from xmltodict.

    Parameters
    ----------
    xml_tag_name : str
        Name of the xml tag to extract from the node.

    node : object
        Node object returned by ``xmltodict`` from which ``xml_tag_name``
        should be extracted.

    allow_none : bool
        If ``False``, the tag needs to exist in the node. Will raise a
        ``ValueError`` if it does not.

    Returns
    -------
    object
    """
    if xml_tag_name in node and node[xml_tag_name] is not None:
        if isinstance(node[xml_tag_name], dict):
            rval = [node[xml_tag_name]]
        elif isinstance(node[xml_tag_name], str):
            rval = [node[xml_tag_name]]
        elif isinstance(node[xml_tag_name], list):
            rval = node[xml_tag_name]
        else:
            raise ValueError('Received not string and non list as tag item')

        return rval
    else:
        if allow_none:
            return None
        else:
            raise ValueError("Could not find tag '%s' in node '%s'" %
                             (xml_tag_name, str(node)))


def _tag_entity(entity_type, entity_id, tag, untag=False):
    """
    Function that tags or untags a given entity on OpenML. As the OpenML
    API tag functions all consist of the same format, this function covers
    all entity types (currently: dataset, task, flow, setup, run). Could
    be used in a partial to provide dataset_tag, dataset_untag, etc.

    Parameters
    ----------
    entity_type : str
        Name of the entity to tag (e.g., run, flow, data)

    entity_id : int
        OpenML id of the entity

    tag : str
        The tag

    untag : bool
        Set to true if needed to untag, rather than tag

    Returns
    -------
    tags : list
        List of tags that the entity is (still) tagged with
    """
    legal_entities = {'data', 'task', 'flow', 'setup', 'run'}
    if entity_type not in legal_entities:
        raise ValueError('Can\'t tag a %s' % entity_type)

    uri = '%s/tag' % entity_type
    main_tag = 'oml:%s_tag' % entity_type
    if untag:
        uri = '%s/untag' % entity_type
        main_tag = 'oml:%s_untag' % entity_type

    post_variables = {'%s_id' % entity_type: entity_id, 'tag': tag}
    result_xml = openml._api_calls._perform_api_call(uri,
                                                     'post',
                                                     post_variables)

    result = xmltodict.parse(result_xml, force_list={'oml:tag'})[main_tag]

    if 'oml:tag' in result:
        return result['oml:tag']
    else:
        # no tags, return empty list
        return []


def _delete_entity(entity_type, entity_id):
    """
    Function that deletes a given entity on OpenML. As the OpenML
    API tag functions all consist of the same format, this function covers
    all entity types that can be deleted (currently: dataset, task, flow,
    run, study and user).

    Parameters
    ----------
    entity_type : str
        Name of the entity to tag (e.g., run, flow, data)

    entity_id : int
        OpenML id of the entity

    Returns
    -------
    bool
        True iff the deletion was successful. False otherwse
    """
    legal_entities = {
        'data',
        'flow',
        'task',
        'run',
        'study',
        'user',
    }
    if entity_type not in legal_entities:
        raise ValueError('Can\'t delete a %s' % entity_type)

    url_suffix = '%s/%d' % (entity_type, entity_id)
    result_xml = openml._api_calls._perform_api_call(url_suffix,
                                                     'delete')
    result = xmltodict.parse(result_xml)
    if 'oml:%s_delete' % entity_type in result:
        return True
    else:
        return False


def _list_all(listing_call, output_format='dict', *args, **filters):
    """Helper to handle paged listing requests.

    Example usage:

    ``evaluations = list_all(list_evaluations, "predictive_accuracy", task=mytask)``

    Parameters
    ----------
    listing_call : callable
        Call listing, e.g. list_evaluations.
    output_format : str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    *args : Variable length argument list
        Any required arguments for the listing call.
    **filters : Arbitrary keyword arguments
        Any filters that can be applied to the listing function.
        additionally, the batch_size can be specified. This is
        useful for testing purposes.
    Returns
    -------
    dict or dataframe
    """

    # eliminate filters that have a None value
    active_filters = {key: value for key, value in filters.items()
                      if value is not None}
    page = 0
    result = {}
    if output_format == 'dataframe':
        result = pd.DataFrame()

    # Default batch size per paging.
    # This one can be set in filters (batch_size), but should not be
    # changed afterwards. The derived batch_size can be changed.
    BATCH_SIZE_ORIG = 10000
    if 'batch_size' in active_filters:
        BATCH_SIZE_ORIG = active_filters['batch_size']
        del active_filters['batch_size']

    # max number of results to be shown
    LIMIT = None
    offset = 0
    if 'size' in active_filters:
        LIMIT = active_filters['size']
        del active_filters['size']

    if LIMIT is not None and BATCH_SIZE_ORIG > LIMIT:
        BATCH_SIZE_ORIG = LIMIT

    if 'offset' in active_filters:
        offset = active_filters['offset']
        del active_filters['offset']

    batch_size = BATCH_SIZE_ORIG
    while True:
        try:
            current_offset = offset + BATCH_SIZE_ORIG * page
            new_batch = listing_call(
                *args,
                limit=batch_size,
                offset=current_offset,
                output_format=output_format,
                **active_filters
            )
        except openml.exceptions.OpenMLServerNoResult:
            # we want to return an empty dict in this case
            break
        if output_format == 'dataframe':
            if len(result) == 0:
                result = new_batch
            else:
                result = result.append(new_batch, ignore_index=True)
        else:
            # For output_format = 'dict' or 'object'
            result.update(new_batch)
        if len(new_batch) < batch_size:
            break
        page += 1
        if LIMIT is not None:
            # check if the number of required results has been achieved
            # always do a 'bigger than' check,
            # in case of bugs to prevent infinite loops
            if len(result) >= LIMIT:
                break
            # check if there are enough results to fulfill a batch
            if BATCH_SIZE_ORIG > LIMIT - len(result):
                batch_size = LIMIT - len(result)

    return result


def _create_cache_directory(key):
    cache = config.get_cache_directory()
    cache_dir = os.path.join(cache, key)
    try:
        os.makedirs(cache_dir)
    except OSError:
        pass
    return cache_dir


def _create_cache_directory_for_id(key, id_):
    """Create the cache directory for a specific ID

    In order to have a clearer cache structure and because every task
    is cached in several files (description, split), there
    is a directory for each task witch the task ID being the directory
    name. This function creates this cache directory.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    key : str

    id_ : int

    Returns
    -------
    str
        Path of the created dataset cache directory.
    """
    cache_dir = os.path.join(
        _create_cache_directory(key), str(id_)
    )
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        pass
    elif os.path.exists(cache_dir) and not os.path.isdir(cache_dir):
        raise ValueError('%s cache dir exists but is not a directory!' % key)
    else:
        os.makedirs(cache_dir)
    return cache_dir


def _remove_cache_dir_for_id(key, cache_dir):
    """Remove the task cache directory

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    key : str

    cache_dir : str
    """
    try:
        shutil.rmtree(cache_dir)
    except (OSError, IOError):
        raise ValueError('Cannot remove faulty %s cache directory %s.'
                         'Please do this manually!' % (key, cache_dir))


def thread_safe_if_oslo_installed(func):
    if oslo_installed:
        @wraps(func)
        def safe_func(*args, **kwargs):
            # Lock directories use the id that is passed as either positional or keyword argument.
            id_parameters = [parameter_name for parameter_name in kwargs if '_id' in parameter_name]
            if len(id_parameters) == 1:
                id_ = kwargs[id_parameters[0]]
            elif len(args) > 0:
                id_ = args[0]
            else:
                raise RuntimeError("An id must be specified for {}, was passed: ({}, {}).".format(
                    func.__name__, args, kwargs
                ))
            # The [7:] gets rid of the 'openml.' prefix
            lock_name = "{}.{}:{}".format(func.__module__[7:], func.__name__, id_)
            with lockutils.external_lock(name=lock_name, lock_path=_create_lockfiles_dir()):
                return func(*args, **kwargs)
        return safe_func
    else:
        return func


def _create_lockfiles_dir():
    dir = os.path.join(config.get_cache_directory(), 'locks')
    try:
        os.makedirs(dir)
    except OSError:
        pass
    return dir


def _download_text_file(source: str,
                        output_path: str,
                        md5_checksum: str = None,
                        exists_ok: bool = True,
                        encoding: str = 'utf8',
                        ) -> None:
    """ Download the text file at `source` and store it in `output_path`.

    By default, do nothing if a file already exists in `output_path`.
    The downloaded file can be checked against an expected md5 checksum.

    Parameters
    ----------
    source : str
        url of the file to be downloaded
    output_path : str
        full path, including filename, of where the file should be stored.
    md5_checksum : str, optional (default=None)
        If not None, should be a string of hexidecimal digits of the expected digest value.
    exists_ok : bool, optional (default=True)
        If False, raise an FileExistsError if there already exists a file at `output_path`.
    encoding : str, optional (default='utf8')
        The encoding with which the file should be stored.
    """
    try:
        with open(output_path, encoding=encoding):
            if exists_ok:
                return
            else:
                raise FileExistsError
    except FileNotFoundError:
        pass

    downloaded_file = openml._api_calls._read_url(source, request_method='get')

    if md5_checksum is not None:
        md5 = hashlib.md5()
        md5.update(downloaded_file.encode('utf-8'))
        md5_checksum_download = md5.hexdigest()
        if md5_checksum != md5_checksum_download:
            raise openml.exceptions.OpenMLHashException(
                'Checksum {} of downloaded file is unequal to the expected checksum {}.'
                .format(md5_checksum_download, md5_checksum))

    with open(output_path, "w", encoding=encoding) as fh:
        fh.write(downloaded_file)

    del downloaded_file
