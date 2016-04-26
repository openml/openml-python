from collections import OrderedDict
import sys
import time

import arff
import numpy as np
import six
import xmltodict

from ..tasks import get_task
from .._api_calls import _perform_api_call


class OpenMLRun(object):
    """OpenML Run: result of running a model on an openml dataset.

    Parameters
    ----------
    FIXME

    """
    def __init__(self, task_id, flow_id, dataset_id, setup_string=None,
                 files=None, setup_id=None, tags=None, run_id=None,
                 uploader=None, uploader_name=None, evaluations=None,
                 detailed_evaluations=None, data_content=None,
                 model=None, task_type=None, task_evaluation_measure=None,
                 flow_name=None, parameter_settings=None, predictions_url=None):
        self.run_id = run_id
        self.uploader = uploader
        self.uploader_name = uploader_name
        self.task_id = task_id
        self.task_type = task_type
        self.task_evaluation_measure = task_evaluation_measure
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.setup_id = setup_id
        self.setup_string = setup_string
        self.parameter_settings = parameter_settings
        self.dataset_id = dataset_id
        self.predictions_url = predictions_url
        self.evaluations = evaluations
        self.detailed_evaluations = detailed_evaluations
        self.data_content = data_content
        self.model = model

    def _generate_arff_header_dict(self):
        """Generates the arff header dictionary for upload to the server.

        Returns
        -------
        arf_dict : dictionary
            Dictionary representation of an ARFF data format containing
            predictions and confidences.
        """
        run_environment = (_get_version_information() +
                           [time.strftime("%c")] + ['Created by run_task()'])
        task = get_task(self.task_id)
        class_labels = task.class_labels

        arff_dict = {}
        arff_dict['attributes'] = [('repeat', 'NUMERIC'),  # lowercase 'numeric' gives an error
                                   ('fold', 'NUMERIC'),
                                   ('row_id', 'NUMERIC')] + \
            [('confidence.' + class_labels[i], 'NUMERIC') for i in range(len(class_labels))] +\
            [('prediction', class_labels),
             ('correct', class_labels)]
        arff_dict['data'] = self.data_content
        arff_dict['description'] = "\n".join(run_environment)
        arff_dict['relation'] = 'openml_task_' + str(task.task_id) + '_predictions'
        return arff_dict

    def publish(self):
        """Publish a run to the OpenML server.

        Uploads the results of a run to OpenML.
        """
        predictions = arff.dumps(self._generate_arff_header_dict())
        description_xml = self._create_description_xml()
        data = {'predictions': ("predictions.csv", predictions),
                'description': ("description.xml", description_xml)}
        return_code, return_value = _perform_api_call(
            "/run/", file_elements=data)
        return return_code, return_value

    def _create_description_xml(self):
        """Create xml representation of run for upload.

        Returns
        -------
        xml_string : string
            XML description of run.
        """
        run_environment = _get_version_information()
        # The setup string is a string necessary to instantiate the model. In
        # this case it's python code necessary to instantiate the passed model
        # with the given hyperparameters!
        setup_string = _create_setup_string(self.model)

        parameter_settings = self.model.get_params()
        # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
        # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
        well_formatted_time = time.strftime("%c").replace(
            ' ', '_').replace('/', '-').replace(':', '.')
        tags = run_environment + [well_formatted_time] + ['run_task'] + \
            [self.model.__module__ + "." + self.model.__class__.__name__]
        description = _to_dict(
            self.task_id, self.flow_id, setup_string, parameter_settings, tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml


def _create_setup_string(model):
    """Create a string representing the model"""
    run_environment = _get_version_information()
    run_environment = "".join(['# %s\n' % package
                               for package in run_environment])
    # fixme str(model) might contain (...)
    return run_environment + _pprint_model(model)


def _pprint_model(model):
    """Pretty print a model

    Copied from sklearn version 0.18 dev.
    """
    class_name = model.__class__.__name__
    return '%s(%s)' % (class_name,
                       _pprint_parameters(model.get_params(deep=False),
                                          offset=len(class_name), ),)


def _pprint_parameters(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Code from sklearn version 0.18 dev, changed to output representation of
    parameters longer than 500 characters

    Parameters
    ----------
    params: dict
        The dictionary to pretty print

    offset: int
        The offset in characters to add at the begin of each line.

    printer:
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


# This can possibly be done by a package such as pyxb, but I could not get
# it to work properly.
def _get_version_information():
    """Gets versions of python, sklearn, numpy and scipy, returns them in an array,

    Returns
    -------
    result : an array with version information of the above packages
    """
    import sklearn
    import scipy
    import numpy

    major, minor, micro, _, _ = sys.version_info
    python_version = 'Python_{}.'.format(
        ".".join([str(major), str(minor), str(micro)]))
    sklearn_version = 'Sklearn_{}.'.format(sklearn.__version__)
    numpy_version = 'NumPy_{}.'.format(numpy.__version__)
    scipy_version = 'SciPy_{}.'.format(scipy.__version__)

    return [python_version, sklearn_version, numpy_version, scipy_version]


def _to_dict(taskid, flow_id, setup_string, parameter_settings, tags):
    """ Creates a dictionary corresponding to the desired xml desired by openML

    Parameters
    ----------
    taskid : int
        the identifier of the task
    setup_string : string
        a CLI string which can invoke the learning with the correct parameter settings
    parameter_settings : array of dicts
        each dict containing keys name, value and component, one per parameter setting
    tags : array of strings
        information that give a description of the run, must conform to
        regex ``([a-zA-Z0-9_\-\.])+``

    Returns
    -------
    result : an array with version information of the above packages
    """
    description = OrderedDict()
    description['oml:run'] = OrderedDict()
    description['oml:run']['@xmlns:oml'] = 'http://openml.org/openml'
    description['oml:run']['oml:task_id'] = taskid
    description['oml:run']['oml:flow_id'] = flow_id

    params = []
    for k, v in parameter_settings.items():
        param_dict = OrderedDict()
        param_dict['oml:name'] = k
        param_dict['oml:value'] = ('None' if v is None else v)
        params.append(param_dict)

    description['oml:run']['oml:parameter_setting'] = params
    description['oml:run']['oml:tag'] = tags  # Tags describing the run
    # description['oml:run']['oml:output_data'] = 0;
    # all data that was output of this run, which can be evaluation scores
    # (though those are also calculated serverside)
    # must be of special data type
    return description
