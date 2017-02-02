from collections import OrderedDict
import sys
import time

import arff
import xmltodict
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV

import openml
from ..tasks import get_task
from .._api_calls import _perform_api_call
from ..exceptions import PyOpenMLError

class OpenMLRun(object):
    """OpenML Run: result of running a model on an openml dataset.

    Parameters
    ----------
    FIXME

    """
    def __init__(self, task_id, flow_id, dataset_id, setup_string=None,
                 files=None, setup_id=None, tags=None, uploader=None, uploader_name=None,
                 evaluations=None, detailed_evaluations=None,
                 data_content=None, trace_content=None, model=None, task_type=None,
                 task_evaluation_measure=None, flow_name=None,
                 parameter_settings=None, predictions_url=None, task=None,
                 flow=None, run_id=None):
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
        self.trace_content = trace_content
        self.task = task
        self.flow = flow
        self.run_id = run_id
        self.model = model

    def _generate_arff_dict(self):
        """Generates the arff dictionary for uploading predictions to the server.

        Assumes that the run has been executed.

        Returns
        -------
        arf_dict : dict
            Dictionary representation of the ARFF file that will be uploaded.
            Contains predictions and information about the run environment.
        """
        if self.data_content is None:
            raise ValueError('Run has not been executed.')

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

    def _generate_trace_arff_dict(self, model):
        """Generates the arff dictionary for uploading predictions to the server.

        Assumes that the run has been executed.

        Returns
        -------
        arf_dict : dict
            Dictionary representation of the ARFF file that will be uploaded.
            Contains information about the optimization trace.
        """
        if self.trace_content is None:
            raise ValueError('No trace content avaiable.')
        if not isinstance(model, BaseSearchCV):
            raise PyOpenMLError('Cannot generate trace on provided classifier. (This should never happen.)')

        arff_dict = {}
        arff_dict['attributes'] = [('repeat', 'NUMERIC'),
                                   ('fold', 'NUMERIC'),
                                   ('iteration', 'NUMERIC'),
                                   ('evaluation', 'NUMERIC'),
                                   ('selected', ['true', 'false'])]
        for key in model.cv_results_:
            if key.startswith("param_"):
                type = 'STRING'
                if all(isinstance(i, (bool)) for i in model.cv_results_[key]):
                    type = ['True', 'False']
                elif all(isinstance(i, (int, float)) for i in model.cv_results_[key]):
                    type = 'NUMERIC'
                else:
                    values = list(set(model.cv_results_[key])) # unique values
                    type = [str(i) for i in values]
                    print(key + ": " + str(type))

                attribute = ("parameter_" + key[6:], type)
                arff_dict['attributes'].append(attribute)

        arff_dict['data'] = self.trace_content
        arff_dict['relation'] = 'openml_task_' + str(self.task_id) + '_predictions'

        return arff_dict

    def publish(self):
        """Publish a run to the OpenML server.

        Uploads the results of a run to OpenML.
        Sets the run_id on self

        Returns
        -------
        self : OpenMLRun
        """
        if self.model is None:
            raise PyOpenMLError("OpenMLRun obj does not contain a model. (This should never happen.) ");
        if self.flow_id is None:
            raise PyOpenMLError("OpenMLRun obj does not contain a flow id. (Should have been added while executing the task.) ");


        predictions = arff.dumps(self._generate_arff_dict())
        description_xml = self._create_description_xml()

        file_elements = {'predictions': ("predictions.arff", predictions),
                         'description': ("description.xml", description_xml)}
        if self.trace_content is not None:
            trace_arff = arff.dumps(self._generate_trace_arff_dict(self.model))
            file_elements['trace'] = ("trace.arff", trace_arff)

        return_value = _perform_api_call("/run/", file_elements=file_elements)
        run_id = int(xmltodict.parse(return_value)['oml:upload_run']['oml:run_id'])
        self.run_id = run_id
        return self

    def _create_description_xml(self):
        """Create xml representation of run for upload.

        Returns
        -------
        xml_string : string
            XML description of run.
        """
        run_environment = _get_version_information()

        # TODO: don't we have flow object in data structure? Use this one
        downloaded_flow = openml.flows.get_flow(self.flow_id)

        openml_param_settings = _parse_parameters(self.model, downloaded_flow)

        # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
        # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
        well_formatted_time = time.strftime("%c").replace(
            ' ', '_').replace('/', '-').replace(':', '.')
        tags = run_environment + [well_formatted_time] + ['run_task'] + \
            [self.model.__module__ + "." + self.model.__class__.__name__]
        description = _to_dict(taskid=self.task_id, flow_id=self.flow_id,
                               setup_string=_create_setup_string(self.model),
                               parameter_settings=openml_param_settings,
                               tags=tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml

def _parse_parameters(model, flow):
    """Extracts all parameter settings from a model in OpenML format.

    Parameters
    ----------
    model
        the scikit-learn model (fitted)
    flow
        openml flow object (containing flow ids, i.e., it has to be downloaded from the server)

    """
    python_param_settings = model.get_params()
    openml_param_settings = []

    def get_flow_dict(_flow):
        flow_map = {_flow.name: _flow.flow_id}
        for subflow in _flow.components:
            flow_map.update(get_flow_dict(_flow.components[subflow]))
        return flow_map

    flow_dict = get_flow_dict(flow)

    for param in python_param_settings:
        if "__" in param:
            # parameter of subflow. will be handled later
            continue
        if isinstance(python_param_settings[param], BaseEstimator):
            # extract parameters of the subflow individually
            subflow = flow.components[param]
            openml_param_settings += _parse_parameters(python_param_settings[param], subflow)

        # add parameter setting (also the subflow. Just because we can)
        param_dict = OrderedDict()
        param_dict['oml:name'] = param
        param_dict['oml:value'] = str(python_param_settings[param])
        param_dict['oml:component'] = flow_dict[flow.name]
        openml_param_settings.append(param_dict)

    return openml_param_settings

################################################################################
# Functions which cannot be in runs/functions due to circular imports


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
    description['oml:run']['oml:parameter_setting'] = parameter_settings
    description['oml:run']['oml:tag'] = tags  # Tags describing the run
    # description['oml:run']['oml:output_data'] = 0;
    # all data that was output of this run, which can be evaluation scores
    # (though those are also calculated serverside)
    # must be of special data type
    return description


def _create_setup_string(model):
    """Create a string representing the model"""
    run_environment = " ".join(_get_version_information())
    # fixme str(model) might contain (...)
    return run_environment + " " + str(model)