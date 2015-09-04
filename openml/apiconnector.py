from collections import OrderedDict
import hashlib
import logging
import os
import re
import sys
import tempfile
import requests
import arff

if sys.version_info[0] < 3:
    import ConfigParser as configparser
    from StringIO import StringIO
    from urllib import urlencode, urlopen
    from urllib2 import URLError, urlopen
else:
    import configparser
    from io import StringIO
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from urllib.error import URLError

import xmltodict

from .entities.dataset import OpenMLDataset
from .entities.task import OpenMLTask
from .entities.split import OpenMLSplit
from .entities.run import OpenMLRun
from .util import is_string

import numpy as np

logger = logging.getLogger(__name__)

OPENML_URL = "http://api_new.openml.org/v1/"


class OpenMLStatusChange(Warning):
    def __init__(self, message):
        super(OpenMLStatusChange, self).__init__(message)


class OpenMLDatasetStatusChange(OpenMLStatusChange):
    def __init__(self, message):
        super(OpenMLDatasetStatusChange, self).__init__(message)


class PyOpenMLError(Exception):
    def __init__(self, message):
        super(PyOpenMLError, self).__init__(message)


class OpenMLServerError(PyOpenMLError):
    def __init__(self, message):
        super(OpenMLServerError, self).__init__(message)


class OpenMLCacheException(PyOpenMLError):
    def __init__(self, message):
        super(OpenMLCacheException, self).__init__(message)


class APIConnector(object):
    """
    Provides an interface to the OpenML server.

    All parameters of the APIConnector can be either specified in a config
    file or when creating this object. The config file must be placed in a
    directory ``.openml`` inside the users home directory and have the name
    ``config``. If one of the parameters is specified by passing it to the
    constructor of this class, it will override the value specified in the
    configuration file.

    Parameters
    ----------
    cache_directory : string, optional (default=None)
        A local directory which will be used for caching. If this is not set, a
        directory '.openml/cache' in the users home directory will be used.
        If either directory does not exist, it will be created.

    apikey : string, optional (default=None)
        Your OpenML API key which will be used to authenticate you at the OpenML
        server.

    server : string, optional (default=None)
        The OpenML server to connect to.

    verbosity : int, optional (default=None)

    configure_logger : bool (default=True)
        Whether the python logging module should be configured by the openml
        package. If set to true, this is a very basic configuration,
        which only prints to the standard output. This is only recommended
        for testing or small problems. It is set to True to adhere to the
        `specifications of the OpenML client API
        <https://github.com/openml/OpenML/wiki/Client-API>`_.
        When the openml module is used as a library, it is recommended that
        the main application controls the logging level, e.g. see
        `here <http://pieces.openpolitics.com
        /2012/04/python-logging-best-practices/>`_.

    private_directory : str, optional (default=None)
        A local directory which can be accessed through the OpenML package.
        Useful to access private datasets through the same interface.

    Raises
    ------
    ValueError
        If apikey is neither specified in the config nor given as an argument.
    OpenMLServerError
        If the OpenML server returns an unexptected response.

    Testing the API calls in Firefox
    --------------------------------
    With the Firefox AddOn HTTPRequestor, one can check the OpenML API calls.

    """
    def __init__(self, cache_directory=None, apikey=None,
                 server=None, verbosity=None, configure_logger=True,
                 private_directory=None):
        # The .openml directory is necessary, just try to create it (EAFP)
        try:
            os.mkdir(os.path.expanduser('~/.openml'))
        except (IOError, OSError):
            # TODO add debug information
            pass

        # Set all variables in the configuration object
        self.config = self._parse_config()
        if cache_directory is not None:
            self.config.set('FAKE_SECTION', 'cachedir', cache_directory)
        if apikey is not None:
            self.config.set('FAKE_SECTION', 'apikey', apikey)
        if server is not None:
            self.config.set('FAKE_SECTION', 'server', server)
        if verbosity is not None:
            self.config.set('FAKE_SECTION', 'verbosity', verbosity)
        if private_directory is not None:
            self.config.set('FAKE_SECTION', 'private_directory', private_directory)

        if configure_logger:
            verbosity = self.config.getint('FAKE_SECTION', 'verbosity')
            if verbosity <= 0:
                level = logging.ERROR
            elif verbosity == 1:
                level = logging.INFO
            elif verbosity >= 2:
                level = logging.DEBUG
            logging.basicConfig(
                format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                       'message)s', datefmt='%H:%M:%S', level=level)

        # Set up the cache directories
        self.cache_dir = self.config.get('FAKE_SECTION', 'cachedir')
        self.dataset_cache_dir = os.path.join(self.cache_dir, "datasets")
        self.task_cache_dir = os.path.join(self.cache_dir, "tasks")

        # Set up the private directory
        self.private_directory = self.config.get('FAKE_SECTION',
                                                 'private_directory')
        self._private_directory_datasets = os.path.join(
            self.private_directory, "datasets")
        self._private_directory_tasks = os.path.join(
            self.private_directory, "tasks")

        for dir_ in [self.cache_dir, self.dataset_cache_dir,
                     self.task_cache_dir, self.private_directory,
                     self._private_directory_datasets,
                     self._private_directory_tasks]:
            if not os.path.exists(dir_) and not os.path.isdir(dir_):
                os.mkdir(dir_)

    def _parse_config(self):
        defaults = {'apikey': '',
                    'server': OPENML_URL,
                    'verbosity': 0,
                    'cachedir': os.path.expanduser('~/.openml/cache'),
                    'private_directory': os.path.expanduser('~/.openml/private')}

        config_file = os.path.expanduser('~/.openml/config')
        config = configparser.RawConfigParser(defaults=defaults)

        if not os.path.exists(config_file):
            # Create an empty config file if there was none so far
            fh = open(config_file, "w")
            fh.close()
            logger.info("Could not find a configuration file at %s. Going to "
                        "create an empty file there." % config_file)

        try:
            # Cheat the ConfigParser module by adding a fake section header
            config_file_ = StringIO()
            config_file_.write("[FAKE_SECTION]\n")
            with open(config_file) as fh:
                for line in fh:
                    config_file_.write(line)
            config_file_.seek(0)
            config.readfp(config_file_)
        except OSError as e:
            logging.info("Error opening file %s: %s" %
                         config_file, e.message)
        return config

    ############################################################################
    # Local getters/accessors to the cache directory
    def get_list_of_cached_datasets(self):
        """Return list with ids of all cached datasets"""
        datasets = []

        for dataset_cache_dir in [self.dataset_cache_dir,
                                  self._private_directory_datasets]:
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

    def get_cached_datasets(self):
        """Searches for all OpenML datasets in the OpenML cache dir.

        Return a dictionary which maps dataset ids to dataset objects"""
        dataset_list = self.get_list_of_cached_datasets()
        datasets = OrderedDict()

        for did in dataset_list:
            datasets[did] = self.get_cached_dataset(did)

        return datasets

    def get_cached_dataset(self, did):
        # This code is slow...replace it with new API calls
        description = self._get_cached_dataset_description(did)
        arff_file = self._get_cached_dataset_arff(did)
        dataset = self._create_dataset_from_description(description, arff_file)

        return dataset

    def _get_cached_dataset_description(self, did):
        for dataset_cache_dir in [self.dataset_cache_dir,
                                  self._private_directory_datasets]:
            did_cache_dir = os.path.join(dataset_cache_dir, str(did))
            description_file = os.path.join(did_cache_dir, "description.xml")

            try:
                with open(description_file) as fh:
                    dataset_xml = fh.read()
            except (IOError, OSError) as e:
                continue

            return xmltodict.parse(dataset_xml)["oml:data_set_description"]

        raise OpenMLCacheException("Dataset description for did %d not "
                                   "cached" % did)


    def _get_cached_dataset_arff(self, did):
        for dataset_cache_dir in [self.dataset_cache_dir,
                                  self._private_directory_datasets]:
            did_cache_dir = os.path.join(dataset_cache_dir, str(did))
            output_file = os.path.join(did_cache_dir, "dataset.arff")

            try:
                with open(output_file):
                    pass
                return output_file
            except (OSError, IOError) as e:
                # TODO create NOTCACHEDEXCEPTION
                continue

        print("Dataset ID", did)
        raise Exception()


    def get_cached_tasks(self):
        tasks = OrderedDict()
        for task_cache_dir in [self.task_cache_dir,
                                  self._private_directory_tasks]:

            directory_content = os.listdir(task_cache_dir)
            directory_content.sort()

            # Find all dataset ids for which we have downloaded the dataset
            # description

            for filename in directory_content:
                match = re.match(r"(tid)_([0-9]*)\.xml", filename)
                if match:
                    tid = match.group(2)
                    tid = int(tid)

                    tasks[tid] = self.get_cached_task(tid)

        return tasks

    def get_cached_task(self, tid):
        for task_cache_dir in [self.task_cache_dir,
                               self._private_directory_tasks]:
            task_file = os.path.join(task_cache_dir,
                                     "tid_%d.xml" % int(tid))

            try:
                with open(task_file) as fh:
                    task = self._create_task_from_xml(xml=fh.read())
                return task
            except (OSError, IOError) as e:
                continue

        print("Task ID", tid)
        raise Exception()

    def get_cached_splits(self):
        splits = OrderedDict()
        for task_cache_dir in [self.task_cache_dir,
                               self._private_directory_tasks]:
            directory_content = os.listdir(task_cache_dir)
            directory_content.sort()


            for filename in directory_content:
                match = re.match(r"(tid)_([0-9]*)\.arff", filename)
                if match:
                    tid = match.group(2)
                    tid = int(tid)

                    splits[tid] = self.get_cached_task(tid)

        return splits

    def get_cached_split(self, tid):
        for task_cache_dir in [self.task_cache_dir,
                               self._private_directory_tasks]:
            try:
                split_file = os.path.join(task_cache_dir,
                                          "tid_%d.arff" % int(tid))
                split = OpenMLSplit.from_arff_file(split_file)
                return split

            except (OSError, IOError) as e:
                continue

        print("Task ID", tid)
        raise Exception()

    ############################################################################
    # Remote getters/API calls to OpenML

    ############################################################################
    # Datasets

    def get_dataset_list(self):
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
        return_code, xml_string = self._perform_api_call("data/list/")
        datasets_dict = xmltodict.parse(xml_string)

        # Minimalistic check if the XML is useful
        assert type(datasets_dict['oml:data']['oml:dataset']) == list, \
            type(datasets_dict['oml:data'])
        assert datasets_dict['oml:data']['@xmlns:oml'] == \
            'http://openml.org/openml'

        datasets = []
        for dataset_ in datasets_dict['oml:data']['oml:dataset']:
            dataset = {'did': int(dataset_['oml:did']),
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

    def datasets_active(self, dids):
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
        dataset_list = self.get_dataset_list()
        dids = sorted(dids)
        active = {}

        dataset_list_idx = 0
        for did in dids:
            # TODO replace with a more efficient while loop!
            for idx in range(dataset_list_idx, len(dataset_list)):
                if did == dataset_list[idx]['did']:
                    active['did'] = bool(dataset_list[idx]['status'])
            dataset_list_idx = idx

    def download_datasets(self, dids):
        """Download datasets.

        Parameters
        ----------
        dids : iterable
            A list of integers representing dataset ids.

        Returns
        -------
        list
            A list of dataset objects.

        Note
        ----
        Uses the method :method:`pyMetaLearn.data_repositories.openml
        .apiconnector.APIConnector.download_dataset` internally. Please read
        the documentation of this.
        """
        datasets = []
        for did in dids:
            datasets.append(self.download_dataset(did))
        return datasets

    def download_dataset(self, did):
        """Download a dataset.

        TODO: explain caching!

        Parameters
        ----------
        dids : int
            Dataset ID of the dataset to download

        Returns
        -------
        dataset : :class:`pyMetaLearn.entities.dataset.Dataset`
            The downloaded dataset."""
        try:
            did = int(did)
        except:
            raise ValueError("Dataset ID is neither an Integer nor can be "
                             "cast to an Integer.")

        description = self.download_dataset_description(did)
        arff_file = self.download_dataset_arff(did, description=description)

        dataset = self._create_dataset_from_description(description, arff_file)
        return dataset

    def download_dataset_description(self, did):
        # TODO implement a cache for this that invalidates itself after some
        # time
        # This can be saved on disk, but cannot be cached properly, because
        # it contains the information on whether a dataset is active.
        did_cache_dir = self._create_dataset_cache_dir(did)
        description_file = os.path.join(did_cache_dir, "description.xml")

        try:
            return self._get_cached_dataset_description(did)
        except (OpenMLCacheException):
            try:
                return_code, dataset_xml = self._perform_api_call(
                    "data/%d" % did)
            except (URLError, UnicodeEncodeError) as e:
                # TODO logger.debug
                self._remove_dataset_chache_dir(did)
                print(e)
                raise e

            with open(description_file, "w") as fh:
                fh.write(dataset_xml)

        try:
            description = xmltodict.parse(dataset_xml)[
                "oml:data_set_description"]
        except Exception as e:
            # TODO logger.debug
            self._remove_dataset_chache_dir(did)
            print("Dataset ID", did)
            raise e

        with open(description_file, "w") as fh:
            fh.write(dataset_xml)

        return description

    def download_dataset_arff(self, did, description=None):
        did_cache_dir = self._create_dataset_cache_dir(did)
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
            description = self.download_dataset_description(did)
        url = description['oml:url']
        return_code, arff_string = self._read_url(url)
        # TODO: it is inefficient to load the dataset in memory prior to
        # saving it to the hard disk!
        with open(output_file, "w") as fh:
            fh.write(arff_string)
        del arff_string

        return output_file

    def download_dataset_features(self, did):
        did_cache_dir = self._create_dataset_cache_dir(did)
        features_file = os.path.join(did_cache_dir, "features.xml")

        # Dataset features aren't subject to change...
        try:
            with open(features_file) as fh:
                features_xml = fh.read()
        except (OSError, IOError):
            try:
                return_code, features_xml = self._perform_api_call(
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

    def download_dataset_qualities(self, did):
        # Dataset qualities are subject to change and must be fetched every time
        did_cache_dir = self._create_dataset_cache_dir(did)
        qualities_file = os.path.join(did_cache_dir, "qualities.xml")
        try:
            return_code, qualities_xml = self._perform_api_call(
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

    def _create_dataset_cache_dir(self, did):
        dataset_cache_dir = os.path.join(self.dataset_cache_dir, str(did))
        try:
            os.makedirs(dataset_cache_dir)
        except (OSError, IOError):
            # TODO add debug information!
            pass
        return dataset_cache_dir

    def _remove_dataset_chache_dir(self, did):
        dataset_cache_dir = os.path.join(self.dataset_cache_dir, str(did))
        try:
            os.rmdir(dataset_cache_dir)
        except (OSError, IOError):
            # TODO add debug information
            pass

    def _create_dataset_from_description(self, description, arff_file):
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

    ############################################################################
    # Tasks
    def get_task_list(self, task_type_id=1):
        """Return a list of all tasks which are on OpenML.

        Parameters
        ----------
        task_type_id : int
            ID of the task type as detailed
            `here <http://openml.org/api/?f=openml.task.types>`_.

        Returns
        -------
        tasks : list
            A list of all tasks. Every task is represented by a
            dictionary containing the following information: task id,
            dataset id, task_type and status. If qualities are calculated for
            the associated dataset, some of these are also returned.
        """
        try:
            task_type_id = int(task_type_id)
        except:
            raise ValueError("Task Type ID is neither an Integer nor can be "
                             "cast to an Integer.")

        return_code, xml_string = self._perform_api_call(
            "task/list/%d" % task_type_id)
        tasks_dict = xmltodict.parse(xml_string)
        # Minimalistic check if the XML is useful
        assert tasks_dict['oml:tasks']['@xmlns:oml'] == \
            'http://openml.org/openml'
        assert type(tasks_dict['oml:tasks']['oml:task']) == list

        tasks = []
        for task_ in tasks_dict['oml:tasks']['oml:task']:
            task = {'tid': int(task_['oml:task_id']),
                    'did': int(task_['oml:did']),
                    'task_type': task_['oml:task_type'],
                    'status': task_['oml:status']}

            # The number of qualities can range from 0 to infinity
            for quality in task_.get('oml:quality', list()):
                quality['#text'] = float(quality['#text'])
                if abs(int(quality['#text']) - quality['#text']) < 0.0000001:
                    quality['#text'] = int(quality['#text'])
                task[quality['@name']] = quality['#text']

            tasks.append(task)
        tasks.sort(key=lambda t: t['tid'])

        return tasks

    def download_task(self, task_id):
        """Download the OpenML task for a given task ID.

        Parameters
        ----------
        task_id : int
            The OpenML task id.
        """
        try:
            task_id = int(task_id)
        except:
            raise ValueError("Task ID is neither an Integer nor can be "
                             "cast to an Integer.")

        xml_file = os.path.join(self._create_task_cache_dir(task_id),
                                "task.xml")

        try:
            with open(xml_file) as fh:
                task = self._create_task_from_xml(fh.read())
        except (OSError, IOError):

            try:
                return_code, task_xml = self._perform_api_call(
                    "task/%d" % task_id)
            except (URLError, UnicodeEncodeError) as e:
                print(e)
                raise e

            # Cache the xml task file
            if os.path.exists(xml_file):
                with open(xml_file) as fh:
                    local_xml = fh.read()

                if task_xml != local_xml:
                    raise ValueError("Task description of task %d cached at %s "
                                     "has changed." % (task_id, xml_file))

            else:
                with open(xml_file, "w") as fh:
                    fh.write(task_xml)

            task = self._create_task_from_xml(task_xml)

        self.download_split(task)
        self.download_dataset(task.dataset_id)
        return task

    def _create_task_from_xml(self, xml):
        dic = xmltodict.parse(xml)["oml:task"]

        estimation_parameters = dict()
        inputs = dict()
        # Due to the unordered structure we obtain, we first have to extract
        # the possible keys of oml:input; dic["oml:input"] is a list of
        # OrderedDicts
        for input_ in dic["oml:input"]:
            name = input_["@name"]
            inputs[name] = input_

        # Convert some more parameters
        for parameter in \
                inputs["estimation_procedure"]["oml:estimation_procedure"][
                    "oml:parameter"]:
            name = parameter["@name"]
            text = parameter.get("#text", "")
            estimation_parameters[name] = text

        return OpenMLTask(
            dic["oml:task_id"], dic["oml:task_type"],
            inputs["source_data"]["oml:data_set"]["oml:data_set_id"],
            inputs["source_data"]["oml:data_set"]["oml:target_feature"],
            inputs["estimation_procedure"]["oml:estimation_procedure"][
                "oml:type"],
            inputs["estimation_procedure"]["oml:estimation_procedure"][
                "oml:data_splits_url"], estimation_parameters,
            inputs["evaluation_measures"]["oml:evaluation_measures"][
                "oml:evaluation_measure"], None, self)

    def download_split(self, task):
        """Download the OpenML split for a given task.

        Parameters
        ----------
        task_id : Task
            An entity of :class:`pyMetaLearn.entities.task.Task`.
        """
        cached_split_file = os.path.join(
            self._create_task_cache_dir(task.task_id), "datasplits.arff")

        try:
            split = OpenMLSplit.from_arff_file(cached_split_file)
        # Add FileNotFoundError in python3 version (which should be a
        # subclass of OSError.
        except (OSError, IOError):
            # Next, download and cache the associated split file
            self._download_split(task, cached_split_file)
            split = OpenMLSplit.from_arff_file(cached_split_file)

        return split

    def _download_split(self, task, cache_file):
        try:
            with open(cache_file):
                pass
        except (OSError, IOError):
            split_url = task.estimation_procedure["data_splits_url"]
            try:
                return_code, split_arff = self._read_url(split_url)
            except (URLError, UnicodeEncodeError) as e:
                print(e, split_url)
                raise e

            with open(cache_file, "w") as fh:
                fh.write(split_arff)
            del split_arff

    def _create_task_cache_dir(self, task_id):
        task_cache_dir = os.path.join(self.task_cache_dir, str(task_id))

        try:
            os.makedirs(task_cache_dir)
        except (IOError, OSError):
            # TODO add debug information!
            pass
        return task_cache_dir

    def _perform_api_call(self, call, data=None, file_dictionary=None, add_authentication=True):
        """
        Perform an API call at the OpenML server.
        return self._read_url(url, data=data, filePath=filePath,
        def _read_url(self, url, add_authentication=False, data=None, filePath=None):

        Parameters
        ----------
        call : str
            The API call. For example data/list
        data : dict (default=None)
            Dictionary containing data which will be sent to the OpenML
            server via a POST request.
        **kwargs
            Further arguments which are appended as GET arguments.

        Returns
        -------
        return_code : int
            HTTP return code
        return_value : str
            Return value of the OpenML server
        """
        url = self.config.get("FAKE_SECTION", "server")
        if not url.endswith("/"):
            url += "/"
        url += call
        return self._read_url(url, data=data, file_dictionary=file_dictionary)

    def _read_url(self, url, data=None, file_dictionary=None):
        if data is None:
            data = {}
        data['api_key'] = self.config.get('FAKE_SECTION', 'apikey')

        if file_dictionary is not None:
            file_elements = {}
            for key, path in file_dictionary.items():
                if os.path.isabs(path) and os.path.exists(path):
                    try:
                        if key is 'dataset':
                            decoder = arff.ArffDecoder()
                            with open(path) as fh:
                                decoder.decode(fh, encode_nominal=True)
                    except:
                        raise ValueError("The file you have provided is not a valid arff file")

                    file_elements[key] = open(path, 'rb')

                else:
                    raise ValueError("File doesn't exist")
            try:
                response = requests.post(url, data=data, files=file_elements)
                return response.status_code, response

            except URLError as error:
                print(error)
        else:
            data = urlencode(data)
            data = data.encode('utf-8')

            CHUNK = 16 * 1024
            string = StringIO()
            connection = urlopen(url, data=data)
            return_code = connection.getcode()
            content_type = connection.info()['Content-Type']
            # TODO maybe switch on the unicode flag!
            match = re.search(r'text/([\w-]*)(; charset=([\w-]*))?', content_type)
            if match:
                if match.groups()[2] is not None:
                    encoding = match.group(3)
                else:
                    encoding = "ascii"
            else:
                # TODO ask JAN why this happens
                logger.warn("Data from %s has content type %s; going to treat "
                            "this as ascii." % (url, content_type))
                encoding = "ascii"

            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
            with tmp as fh:
                while True:
                    chunk = connection.read(CHUNK)
                    # Chunk is now a proper string (UTF-8 in python)
                    chunk = chunk.decode(encoding)
                    if not chunk:
                        break
                    fh.write(chunk)

            tmp = open(tmp.name, "r")
            with tmp as fh:
                while True:
                    chunk = fh.read(CHUNK)
                    if not chunk:
                        break
                    string.write(chunk)
            return return_code, string.getvalue()

    def upload_dataset(self, description, file_path=None):
        try:
            data = {'description': description}
            if file_path is not None:
                return_code, dataset_xml = self._perform_api_call("/data/",data=data, file_dictionary={'dataset': file_path})
            else:
                return_code, dataset_xml = self._perform_api_call("/data/",data=data)

        except URLError as e:
            # TODO logger.debug
            print(e)
            raise e
        return return_code, dataset_xml

    def upload_flow(self, description, file_path=None):
        try:
            data = {'description': description}
            return_code, dataset_xml = self._perform_api_call("/flow/", data=data, file_dictionary={'source': file_path})

        except URLError as e:
            # TODO logger.debug
            print(e)
            raise e
        return return_code, dataset_xml

    def upload_run(self, files):
        file_dictionary = {}
        if 'predictions' in files:
            try:
                for key, value in files.items():
                    file_dictionary[key] = value

                return_code, dataset_xml = self._perform_api_call("/run/", file_dictionary=file_dictionary)

            except URLError as e:
                # TODO logger.debug
                print(e)
                raise e
            return return_code, dataset_xml
        else:
            raise ValueError("prediction files doesn't exist")
