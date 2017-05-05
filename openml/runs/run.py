from collections import OrderedDict
import sys
import time

import arff
import xmltodict

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
                 output_files=None, setup_id=None, tags=None, uploader=None, uploader_name=None,
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
        self.evaluations = evaluations
        self.detailed_evaluations = detailed_evaluations
        self.data_content = data_content
        self.output_files = output_files
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

        # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
        # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
        # well_formatted_time = time.strftime("%c").replace(
        #     ' ', '_').replace('/', '-').replace(':', '.')
        # tags = run_environment + [well_formatted_time] + ['run_task'] + \
        #     [self.model.__module__ + "." + self.model.__class__.__name__]
        description = _to_dict(taskid=self.task_id, flow_id=self.flow_id,
                               setup_string=_create_setup_string(self.model),
                               parameter_settings=openml_param_settings,
                               error_message=self.error_message,
                               detailed_evaluations=self.detailed_evaluations,
                               tags=self.tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml

    @staticmethod
    def _parse_parameters(model, server_flow):
        """Extracts all parameter settings from a model in OpenML format.

        Parameters
        ----------
        model
            the scikit-learn model (fitted)
        flow
            openml flow object (containing flow ids, i.e., it has to be downloaded from the server)

        """
        if server_flow.flow_id is None:
            raise ValueError("The flow parameter needs to be downloaded from server")

        def get_flow_dict(_flow):
            flow_map = {_flow.name: _flow.flow_id}
            for subflow in _flow.components:
                flow_map.update(get_flow_dict(_flow.components[subflow]))
            return flow_map

        def extract_parameters(_flow, _param_dict, _main_call=False, main_id=None):
            # _flow is openml flow object, _param dict maps from flow name to flow id
            # for the main call, the param dict can be overridden (useful for unit tests / sentinels)
            # this way, for flows without subflows we do not have to rely on _param_dict
            _params = []
            for _param_name in _flow.parameters:
                _current = OrderedDict()
                _current['oml:name'] = _param_name
                _current['oml:value'] = _flow.parameters[_param_name]
                if _main_call:
                    _current['oml:component'] = main_id
                else:
                    _current['oml:component'] = _param_dict[_flow.name]
                _params.append(_current)
            for _identifier in _flow.components:
                _params.extend(extract_parameters(_flow.components[_identifier], _param_dict))
            return _params

        flow_dict = get_flow_dict(server_flow)
        local_flow = openml.flows.sklearn_to_flow(model)

        parameters = extract_parameters(local_flow, flow_dict, True, server_flow.flow_id)
        return parameters

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


def _to_dict(taskid, flow_id, setup_string, error_message, parameter_settings, tags=None, detailed_evaluations=None):
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
    if tags is not None:
        description['oml:run']['oml:tag'] = tags  # Tags describing the run
    if detailed_evaluations is not None:
        description['oml:run']['oml:output_data'] = dict()
        description['oml:run']['oml:output_data']['oml:evaluation'] = list()
        for measure in detailed_evaluations:
            for repeat in detailed_evaluations[measure]:
                for fold, value in detailed_evaluations[measure][repeat].items():
                    current = OrderedDict([('@repeat', str(repeat)), ('@fold', str(fold)),
                                           ('oml:name', measure), ('oml:value', str(value))])
                    description['oml:run']['oml:output_data']['oml:evaluation'].append(current)
    return description


def _create_setup_string(model):
    """Create a string representing the model"""
    run_environment = " ".join(_get_version_information())
    # fixme str(model) might contain (...)
    return run_environment + " " + str(model)