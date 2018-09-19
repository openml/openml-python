import io
import os
from collections import OrderedDict

from .. import config
from .. import datasets
from .split import OpenMLSplit
import openml._api_calls
from ..utils import _create_cache_directory_for_id


class OpenMLTask(object):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 target_name, estimation_procedure_type, data_splits_url,
                 estimation_parameters, evaluation_measure, cost_matrix,
                 class_labels=None):
        self.task_id = int(task_id)
        self.task_type_id = int(task_type_id)
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.target_name = target_name
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procedure_type
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure["parameters"] = estimation_parameters
        self.estimation_parameters = estimation_parameters
        self.evaluation_measure = evaluation_measure
        self.cost_matrix = cost_matrix
        self.class_labels = class_labels
        self.split = None

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")

    def get_dataset(self):
        """Download dataset associated with task"""
        return datasets.get_dataset(self.dataset_id)

    def get_X_and_y(self):
        """Get data associated with the current task.
        
        Returns
        -------
        tuple - X and y

        """
        dataset = self.get_dataset()
        if self.task_type_id not in (1, 2, 3):
            raise NotImplementedError(self.task_type)
        X_and_y = dataset.get_data(target=self.target_name)
        return X_and_y

    def get_train_test_split_indices(self, fold=0, repeat=0, sample=0):
        # Replace with retrieve from cache
        if self.split is None:
            self.split = self.download_split()

        train_indices, test_indices = self.split.get(repeat=repeat, fold=fold, sample=sample)
        return train_indices, test_indices

    def _download_split(self, cache_file):
        try:
            with io.open(cache_file, encoding='utf8'):
                pass
        except (OSError, IOError):
            split_url = self.estimation_procedure["data_splits_url"]
            split_arff = openml._api_calls._read_url(split_url)

            with io.open(cache_file, "w", encoding='utf8') as fh:
                fh.write(split_arff)
            del split_arff

    def download_split(self):
        """Download the OpenML split for a given task.
        """
        cached_split_file = os.path.join(
            _create_cache_directory_for_id('tasks', self.task_id),
            "datasplits.arff",
        )

        try:
            split = OpenMLSplit._from_arff_file(cached_split_file)
        except (OSError, IOError):
            # Next, download and cache the associated split file
            self._download_split(cached_split_file)
            split = OpenMLSplit._from_arff_file(cached_split_file)

        return split

    def get_split_dimensions(self):
        if self.split is None:
            self.split = self.download_split()

        return self.split.repeats, self.split.folds, self.split.samples

    def push_tag(self, tag):
        """Annotates this task with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the task.
        """
        data = {'task_id': self.task_id, 'tag': tag}
        openml._api_calls._perform_api_call("/task/tag", data=data)

    def remove_tag(self, tag):
        """Removes a tag from this task on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the task.
        """
        data = {'task_id': self.task_id, 'tag': tag}
        openml._api_calls._perform_api_call("/task/untag", data=data)


    def _to_dict(self):
        """ Helper function used by _to_xml and itself.

        Creates a dictionary representation of self which can be serialized
        to xml by the function _to_xml. Since a flow can contain subflows
        (components) this helper function calls itself recursively to also
        serialize these flows to dictionaries.

        Uses OrderedDict to ensure consistent ordering when converting to xml.
        The return value (OrderedDict) will be used to create the upload xml
        file. The xml file must have the tags in exactly the order given in the
        xsd schema of a flow (see class docstring).

        Returns
        -------
        OrderedDict
            Flow represented as OrderedDict.

        """
        task_container = OrderedDict()
        task_dict = OrderedDict([('@xmlns:oml', 'http://openml.org/openml')])
        task_container['oml:task'] = task_dict
        _add_if_nonempty(task_dict, 'oml:task_id', self.task_id)

        if getattr(self, "task_type_id") is None:
                raise ValueError("task_type_id is required but None")
        else:
            task_dict["task_type_id"] = self.task_type_id

        for attribute in ["source_data", "target_name"]:

            _add_if_nonempty(task_dict, 'oml:{}'.format(attribute),
                             getattr(self, attribute))

        flow_parameters = []
        for key in self.parameters:
            param_dict = OrderedDict()
            param_dict['oml:name'] = key
            meta_info = self.parameters_meta_info[key]

            _add_if_nonempty(param_dict, 'oml:data_type',
                             meta_info['data_type'])
            param_dict['oml:default_value'] = self.parameters[key]
            _add_if_nonempty(param_dict, 'oml:description',
                             meta_info['description'])

            for key_, value in param_dict.items():
                if key_ is not None and not isinstance(key_, six.string_types):
                    raise ValueError('Parameter name %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (key_, type(key_)))
                if value is not None and not isinstance(value, six.string_types):
                    raise ValueError('Parameter value %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (value, type(value)))

            flow_parameters.append(param_dict)

        flow_dict['oml:parameter'] = flow_parameters

        components = []
        for key in self.components:
            component_dict = OrderedDict()
            component_dict['oml:identifier'] = key
            component_dict['oml:flow'] = \
                self.components[key]._to_dict()['oml:flow']

            for key_ in component_dict:
                # We only need to check if the key is a string, because the
                # value is a flow. The flow itself is valid by recursion
                if key_ is not None and not isinstance(key_, six.string_types):
                    raise ValueError('Parameter name %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (key_, type(key_)))

            components.append(component_dict)

        flow_dict['oml:component'] = components
        flow_dict['oml:tag'] = self.tags
        for attribute in ["binary_url", "binary_format", "binary_md5"]:
            _add_if_nonempty(flow_dict, 'oml:{}'.format(attribute),
                             getattr(self, attribute))

        return flow_container


def _create_task_cache_dir(task_id):
    task_cache_dir = os.path.join(config.get_cache_directory(), "tasks", str(task_id))

    try:
        os.makedirs(task_cache_dir)
    except (IOError, OSError):
        # TODO add debug information!
        pass
    return task_cache_dir


def _add_if_nonempty(dic, key, value):
    if value is not None:
        dic[key] = value