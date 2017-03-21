from collections import OrderedDict
import sys
import time

import arff
import xmltodict
from sklearn.base import BaseEstimator

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
                 data_content=None, trace_attributes=None, trace_content=None,
                 model=None, task_type=None, task_evaluation_measure=None, flow_name=None,
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
        self.trace_attributes = trace_attributes
        self.trace_content = trace_content
        self.error_message = None
        self.task = task
        self.flow = flow
        self.run_id = run_id
        self.model = model
        self.tags = tags

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

    def _generate_trace_arff_dict(self):
        """Generates the arff dictionary for uploading predictions to the server.

        Assumes that the run has been executed.

        Returns
        -------
        arf_dict : dict
            Dictionary representation of the ARFF file that will be uploaded.
            Contains information about the optimization trace.
        """
        if self.trace_content is None or len(self.trace_content) == 0:
            raise ValueError('No trace content avaiable.')
        if len(self.trace_attributes) != len(self.trace_content[0]):
            raise ValueError('Trace_attributes and trace_content not compatible')

        arff_dict = {}
        arff_dict['attributes'] = self.trace_attributes
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

        description_xml = self._create_description_xml()
        file_elements = {'description': ("description.xml", description_xml)}

        if self.error_message is None:
            predictions = arff.dumps(self._generate_arff_dict())
            file_elements['predictions'] = ("predictions.arff", predictions)

        if self.trace_content is not None:
            trace_arff = arff.dumps(self._generate_trace_arff_dict())
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

        # TODO: don't we have flow object in data structure? Use this one
        downloaded_flow = openml.flows.get_flow(self.flow_id)

        openml_param_settings = OpenMLRun._parse_parameters(self.model, downloaded_flow)

        description = _to_dict(taskid=self.task_id, flow_id=self.flow_id,
                               setup_string=_create_setup_string(self.model),
                               parameter_settings=openml_param_settings,
                               error_message=self.error_message,
                               tags=self.tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml

    @staticmethod
    def _parse_parameters(model, flow):
        """Extracts all parameter settings from a model in OpenML format.

        Parameters
        ----------
        model
            the scikit-learn model (fitted)
        flow
            openml flow object (containing flow ids, i.e., it has to be downloaded from the server)

        """
        if flow.flow_id is None:
            raise ValueError("The flow parameter needs to be downloaded from server")

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
                openml_param_settings += OpenMLRun._parse_parameters(python_param_settings[param], subflow)

            # add parameter setting (in some cases also the subflow. Just because we can)
            if param in flow.parameters.keys():
                param_dict = OrderedDict()
                param_dict['oml:name'] = param
                param_dict['oml:value'] = str(python_param_settings[param])
                param_dict['oml:component'] = flow_dict[flow.name]
                openml_param_settings.append(param_dict)
            else:
                if flow.name.startswith("sklearn.pipeline.Pipeline"):
                    # tolerate
                    pass
                elif flow.name.startswith("sklearn.pipeline.FeatureUnion"):
                    # tolerate
                    pass
                elif flow.name.startswith("sklearn.ensemble.voting_classifier.VotingClassifier"):
                    # tolerate
                    pass
                else:
                    raise ValueError("parameter %s not in flow description of flow %s" %(param,flow.name))

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


def _to_dict(taskid, flow_id, setup_string, error_message, parameter_settings, tags):
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
    if error_message is not None:
        description['oml:run']['oml:error_message'] = error_message
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