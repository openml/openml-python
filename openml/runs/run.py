from collections import OrderedDict
import pickle
import time
from typing import Any, IO, TextIO  # noqa F401
import os

import arff
import numpy as np
import xmltodict

import openml
import openml._api_calls
from ..exceptions import PyOpenMLError
from ..flows import get_flow
from ..tasks import (get_task,
                     TaskTypeEnum,
                     OpenMLClassificationTask,
                     OpenMLLearningCurveTask,
                     OpenMLClusteringTask,
                     OpenMLRegressionTask
                     )
from ..utils import _tag_entity


class OpenMLRun(object):
    """OpenML Run: result of running a model on an openml dataset.

       Parameters
       ----------
       task_id : int
           Refers to the task.
       flow_id : int
           Refers to the flow.
       dataset_id: int
           Refers to the data.
    """

    def __init__(self, task_id, flow_id, dataset_id, setup_string=None,
                 output_files=None, setup_id=None, tags=None, uploader=None,
                 uploader_name=None, evaluations=None, fold_evaluations=None,
                 sample_evaluations=None, data_content=None, trace=None,
                 model=None, task_type=None, task_evaluation_measure=None,
                 flow_name=None, parameter_settings=None, predictions_url=None,
                 task=None, flow=None, run_id=None):
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
        self.fold_evaluations = fold_evaluations
        self.sample_evaluations = sample_evaluations
        self.data_content = data_content
        self.output_files = output_files
        self.trace = trace
        self.error_message = None
        self.task = task
        self.flow = flow
        self.run_id = run_id
        self.model = model
        self.tags = tags
        self.predictions_url = predictions_url

    def __str__(self):
        flow_name = self.flow_name
        if flow_name is not None and len(flow_name) > 26:
            # long enough to show sklearn.pipeline.Pipeline
            flow_name = flow_name[:26] + "..."
        return "[run id: {}, task id: {}, flow id: {}, flow name: {}]".format(
            self.run_id, self.task_id, self.flow_id, flow_name)

    def _repr_pretty_(self, pp, cycle):
        pp.text(str(self))

    @classmethod
    def from_filesystem(cls, directory: str, expect_model: bool = True) -> 'OpenMLRun':
        """
        The inverse of the to_filesystem method. Instantiates an OpenMLRun
        object based on files stored on the file system.

        Parameters
        ----------
        directory : str
            a path leading to the folder where the results
            are stored

        expect_model : bool
            if True, it requires the model pickle to be present, and an error
            will be thrown if not. Otherwise, the model might or might not
            be present.

        Returns
        -------
        run : OpenMLRun
            the re-instantiated run object
        """

        # Avoiding cyclic imports
        import openml.runs.functions

        if not os.path.isdir(directory):
            raise ValueError('Could not find folder')

        description_path = os.path.join(directory, 'description.xml')
        predictions_path = os.path.join(directory, 'predictions.arff')
        trace_path = os.path.join(directory, 'trace.arff')
        model_path = os.path.join(directory, 'model.pkl')

        if not os.path.isfile(description_path):
            raise ValueError('Could not find description.xml')
        if not os.path.isfile(predictions_path):
            raise ValueError('Could not find predictions.arff')
        if not os.path.isfile(model_path) and expect_model:
            raise ValueError('Could not find model.pkl')

        with open(description_path, 'r') as fht:
            xml_string = fht.read()
        run = openml.runs.functions._create_run_from_xml(xml_string, from_server=False)

        if run.flow_id is None:
            flow = openml.flows.OpenMLFlow.from_filesystem(directory)
            run.flow = flow
            run.flow_name = flow.name

        with open(predictions_path, 'r') as fht:
            predictions = arff.load(fht)
            run.data_content = predictions['data']

        if os.path.isfile(model_path):
            # note that it will load the model if the file exists, even if
            # expect_model is False
            with open(model_path, 'rb') as fhb:
                run.model = pickle.load(fhb)

        if os.path.isfile(trace_path):
            run.trace = openml.runs.OpenMLRunTrace._from_filesystem(trace_path)

        return run

    def to_filesystem(
        self,
        directory: str,
        store_model: bool = True,
    ) -> None:
        """
        The inverse of the from_filesystem method. Serializes a run
        on the filesystem, to be uploaded later.

        Parameters
        ----------
        directory : str
            a path leading to the folder where the results
            will be stored. Should be empty

        store_model : bool, optional (default=True)
            if True, a model will be pickled as well. As this is the most
            storage expensive part, it is often desirable to not store the
            model.
        """
        if self.data_content is None or self.model is None:
            raise ValueError('Run should have been executed (and contain '
                             'model / predictions)')

        os.makedirs(directory, exist_ok=True)
        if not os.listdir(directory) == []:
            raise ValueError(
                'Output directory {} should be empty'.format(os.path.abspath(directory))
            )

        run_xml = self._create_description_xml()
        predictions_arff = arff.dumps(self._generate_arff_dict())

        # It seems like typing does not allow to define the same variable multiple times
        with open(os.path.join(directory, 'description.xml'), 'w') as fh:  # type: TextIO
            fh.write(run_xml)
        with open(os.path.join(directory, 'predictions.arff'), 'w') as fh:
            fh.write(predictions_arff)
        if store_model:
            with open(os.path.join(directory, 'model.pkl'), 'wb') as fh_b:  # type: IO[bytes]
                pickle.dump(self.model, fh_b)

        if self.flow_id is None:
            self.flow.to_filesystem(directory)

        if self.trace is not None:
            self.trace._to_filesystem(directory)

    def _generate_arff_dict(self) -> 'OrderedDict[str, Any]':
        """Generates the arff dictionary for uploading predictions to the
        server.

        Assumes that the run has been executed.

        Returns
        -------
        arf_dict : dict
            Dictionary representation of the ARFF file that will be uploaded.
            Contains predictions and information about the run environment.
        """
        if self.data_content is None:
            raise ValueError('Run has not been executed.')
        if self.flow is None:
            self.flow = get_flow(self.flow_id)

        run_environment = (self.flow.extension.get_version_information()
                           + [time.strftime("%c")]
                           + ['Created by run_task()'])
        task = get_task(self.task_id)

        arff_dict = OrderedDict()  # type: 'OrderedDict[str, Any]'
        arff_dict['data'] = self.data_content
        arff_dict['description'] = "\n".join(run_environment)
        arff_dict['relation'] =\
            'openml_task_{}_predictions'.format(task.task_id)

        if isinstance(task, OpenMLLearningCurveTask):
            class_labels = task.class_labels
            instance_specifications = [
                ('repeat', 'NUMERIC'),
                ('fold', 'NUMERIC'),
                ('sample', 'NUMERIC'),
                ('row_id', 'NUMERIC')
            ]

            arff_dict['attributes'] = instance_specifications
            if class_labels is not None:
                arff_dict['attributes'] = arff_dict['attributes'] + \
                    [('confidence.' + class_labels[i],
                      'NUMERIC')
                     for i in range(len(class_labels))] + \
                    [('prediction', class_labels),
                     ('correct', class_labels)]
            else:
                raise ValueError('The task has no class labels')

        elif isinstance(task, OpenMLClassificationTask):
            class_labels = task.class_labels
            instance_specifications = [('repeat', 'NUMERIC'),
                                       ('fold', 'NUMERIC'),
                                       ('sample', 'NUMERIC'),  # Legacy
                                       ('row_id', 'NUMERIC')]

            arff_dict['attributes'] = instance_specifications
            if class_labels is not None:
                prediction_confidences = [('confidence.' + class_labels[i],
                                           'NUMERIC')
                                          for i in range(len(class_labels))]
                prediction_and_true = [('prediction', class_labels),
                                       ('correct', class_labels)]
                arff_dict['attributes'] = arff_dict['attributes'] + \
                    prediction_confidences + \
                    prediction_and_true
            else:
                raise ValueError('The task has no class labels')

        elif isinstance(task, OpenMLRegressionTask):
            arff_dict['attributes'] = [('repeat', 'NUMERIC'),
                                       ('fold', 'NUMERIC'),
                                       ('row_id', 'NUMERIC'),
                                       ('prediction', 'NUMERIC'),
                                       ('truth', 'NUMERIC')]

        elif isinstance(task, OpenMLClusteringTask):
            arff_dict['attributes'] = [('repeat', 'NUMERIC'),
                                       ('fold', 'NUMERIC'),
                                       ('row_id', 'NUMERIC'),
                                       ('cluster', 'NUMERIC')]

        else:
            raise NotImplementedError(
                'Task type %s is not yet supported.' % str(task.task_type)
            )

        return arff_dict

    def get_metric_fn(self, sklearn_fn, kwargs=None):
        """Calculates metric scores based on predicted values. Assumes the
        run has been executed locally (and contains run_data). Furthermore,
        it assumes that the 'correct' or 'truth' attribute is specified in
        the arff (which is an optional field, but always the case for
        openml-python runs)

        Parameters
        ----------
        sklearn_fn : function
            a function pointer to a sklearn function that
            accepts ``y_true``, ``y_pred`` and ``**kwargs``

        Returns
        -------
        scores : list
            a list of floats, of length num_folds * num_repeats
        """
        kwargs = kwargs if kwargs else dict()
        if self.data_content is not None and self.task_id is not None:
            predictions_arff = self._generate_arff_dict()
        elif 'predictions' in self.output_files:
            predictions_file_url = openml._api_calls._file_id_to_url(
                self.output_files['predictions'], 'predictions.arff',
            )
            response = openml._api_calls._read_url(predictions_file_url,
                                                   request_method='get')
            predictions_arff = arff.loads(response)
            # TODO: make this a stream reader
        else:
            raise ValueError('Run should have been locally executed or '
                             'contain outputfile reference.')

        # Need to know more about the task to compute scores correctly
        task = get_task(self.task_id)

        attribute_names = [att[0] for att in predictions_arff['attributes']]
        if (task.task_type_id in [TaskTypeEnum.SUPERVISED_CLASSIFICATION,
                                  TaskTypeEnum.LEARNING_CURVE]
                and 'correct' not in attribute_names):
            raise ValueError('Attribute "correct" should be set for '
                             'classification task runs')
        if (task.task_type_id == TaskTypeEnum.SUPERVISED_REGRESSION
                and 'truth' not in attribute_names):
            raise ValueError('Attribute "truth" should be set for '
                             'regression task runs')
        if (task.task_type_id != TaskTypeEnum.CLUSTERING
                and 'prediction' not in attribute_names):
            raise ValueError('Attribute "predict" should be set for '
                             'supervised task runs')

        def _attribute_list_to_dict(attribute_list):
            # convenience function: Creates a mapping to map from the name of
            # attributes present in the arff prediction file to their index.
            # This is necessary because the number of classes can be different
            # for different tasks.
            res = OrderedDict()
            for idx in range(len(attribute_list)):
                res[attribute_list[idx][0]] = idx
            return res

        attribute_dict = \
            _attribute_list_to_dict(predictions_arff['attributes'])

        repeat_idx = attribute_dict['repeat']
        fold_idx = attribute_dict['fold']
        predicted_idx = attribute_dict['prediction']  # Assume supervised task

        if task.task_type_id == TaskTypeEnum.SUPERVISED_CLASSIFICATION or \
                task.task_type_id == TaskTypeEnum.LEARNING_CURVE:
            correct_idx = attribute_dict['correct']
        elif task.task_type_id == TaskTypeEnum.SUPERVISED_REGRESSION:
            correct_idx = attribute_dict['truth']
        has_samples = False
        if 'sample' in attribute_dict:
            sample_idx = attribute_dict['sample']
            has_samples = True

        if predictions_arff['attributes'][predicted_idx][1] != \
                predictions_arff['attributes'][correct_idx][1]:
            pred = predictions_arff['attributes'][predicted_idx][1]
            corr = predictions_arff['attributes'][correct_idx][1]
            raise ValueError('Predicted and Correct do not have equal values:'
                             ' %s Vs. %s' % (str(pred), str(corr)))

        # TODO: these could be cached
        values_predict = {}
        values_correct = {}
        for line_idx, line in enumerate(predictions_arff['data']):
            rep = line[repeat_idx]
            fold = line[fold_idx]
            if has_samples:
                samp = line[sample_idx]
            else:
                samp = 0  # No learning curve sample, always 0

            if task.task_type_id in [TaskTypeEnum.SUPERVISED_CLASSIFICATION,
                                     TaskTypeEnum.LEARNING_CURVE]:
                prediction = predictions_arff['attributes'][predicted_idx][
                    1].index(line[predicted_idx])
                correct = predictions_arff['attributes'][predicted_idx][1]. \
                    index(line[correct_idx])
            elif task.task_type_id == TaskTypeEnum.SUPERVISED_REGRESSION:
                prediction = line[predicted_idx]
                correct = line[correct_idx]
            if rep not in values_predict:
                values_predict[rep] = OrderedDict()
                values_correct[rep] = OrderedDict()
            if fold not in values_predict[rep]:
                values_predict[rep][fold] = OrderedDict()
                values_correct[rep][fold] = OrderedDict()
            if samp not in values_predict[rep][fold]:
                values_predict[rep][fold][samp] = []
                values_correct[rep][fold][samp] = []

            values_predict[rep][fold][samp].append(prediction)
            values_correct[rep][fold][samp].append(correct)

        scores = []
        for rep in values_predict.keys():
            for fold in values_predict[rep].keys():
                last_sample = len(values_predict[rep][fold]) - 1
                y_pred = values_predict[rep][fold][last_sample]
                y_true = values_correct[rep][fold][last_sample]
                scores.append(sklearn_fn(y_true, y_pred, **kwargs))
        return np.array(scores)

    def publish(self) -> 'OpenMLRun':
        """ Publish a run (and if necessary, its flow) to the OpenML server.

        Uploads the results of a run to OpenML.
        If the run is of an unpublished OpenMLFlow, the flow will be uploaded too.
        Sets the run_id on self.

        Returns
        -------
        self : OpenMLRun
        """
        if self.model is None:
            raise PyOpenMLError(
                "OpenMLRun obj does not contain a model. "
                "(This should never happen.) "
            )
        if self.flow_id is None:
            if self.flow is None:
                raise PyOpenMLError(
                    "OpenMLRun object does not contain a flow id or reference to OpenMLFlow "
                    "(these should have been added while executing the task). "
                )
            else:
                # publish the linked Flow before publishing the run.
                self.flow.publish()
                self.flow_id = self.flow.flow_id

        if self.parameter_settings is None:
            if self.flow is None:
                self.flow = openml.flows.get_flow(self.flow_id)
            self.parameter_settings = self.flow.extension.obtain_parameter_values(
                self.flow,
                self.model,
            )

        description_xml = self._create_description_xml()
        file_elements = {'description': ("description.xml", description_xml)}

        if self.error_message is None:
            predictions = arff.dumps(self._generate_arff_dict())
            file_elements['predictions'] = ("predictions.arff", predictions)

        if self.trace is not None:
            trace_arff = arff.dumps(self.trace.trace_to_arff())
            file_elements['trace'] = ("trace.arff", trace_arff)

        return_value = openml._api_calls._perform_api_call(
            "/run/", 'post', file_elements=file_elements
        )
        result = xmltodict.parse(return_value)
        self.run_id = int(result['oml:upload_run']['oml:run_id'])
        return self

    def _create_description_xml(self):
        """Create xml representation of run for upload.

        Returns
        -------
        xml_string : string
            XML description of run.
        """

        # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
        # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
        # well_formatted_time = time.strftime("%c").replace(
        #     ' ', '_').replace('/', '-').replace(':', '.')
        # tags = run_environment + [well_formatted_time] + ['run_task'] + \
        #     [self.model.__module__ + "." + self.model.__class__.__name__]
        description = _to_dict(taskid=self.task_id, flow_id=self.flow_id,
                               setup_string=self.setup_string,
                               parameter_settings=self.parameter_settings,
                               error_message=self.error_message,
                               fold_evaluations=self.fold_evaluations,
                               sample_evaluations=self.sample_evaluations,
                               tags=self.tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml

    def push_tag(self, tag: str) -> None:
        """Annotates this run with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the run.
        """
        _tag_entity('run', self.run_id, tag)

    def remove_tag(self, tag: str) -> None:
        """Removes a tag from this run on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the run.
        """
        _tag_entity('run', self.run_id, tag, untag=True)


###############################################################################
# Functions which cannot be in runs/functions due to circular imports

def _to_dict(taskid, flow_id, setup_string, error_message, parameter_settings,
             tags=None, fold_evaluations=None, sample_evaluations=None):
    """ Creates a dictionary corresponding to the desired xml desired by openML

    Parameters
    ----------
    taskid : int
        the identifier of the task
    setup_string : string
        a CLI string which can invoke the learning with the correct parameter
        settings
    parameter_settings : array of dicts
        each dict containing keys name, value and component, one per parameter
        setting
    tags : array of strings
        information that give a description of the run, must conform to
        regex ``([a-zA-Z0-9_\-\.])+``
    fold_evaluations : dict mapping from evaluation measure to a dict mapping
        repeat_nr to a dict mapping from fold nr to a value (double)
    sample_evaluations : dict mapping from evaluation measure to a dict
        mapping repeat_nr to a dict mapping from fold nr to a dict mapping to
        a sample nr to a value (double)
    sample_evaluations :
    Returns
    -------
    result : an array with version information of the above packages
    """  # noqa: W605
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
    if (fold_evaluations is not None and len(fold_evaluations) > 0) or \
            (sample_evaluations is not None and len(sample_evaluations) > 0):
        description['oml:run']['oml:output_data'] = OrderedDict()
        description['oml:run']['oml:output_data']['oml:evaluation'] = list()
    if fold_evaluations is not None:
        for measure in fold_evaluations:
            for repeat in fold_evaluations[measure]:
                for fold, value in fold_evaluations[measure][repeat].items():
                    current = OrderedDict([
                        ('@repeat', str(repeat)), ('@fold', str(fold)),
                        ('oml:name', measure), ('oml:value', str(value))])
                    description['oml:run']['oml:output_data'][
                        'oml:evaluation'].append(current)
    if sample_evaluations is not None:
        for measure in sample_evaluations:
            for repeat in sample_evaluations[measure]:
                for fold in sample_evaluations[measure][repeat]:
                    for sample, value in sample_evaluations[measure][repeat][
                            fold].items():
                        current = OrderedDict([
                            ('@repeat', str(repeat)), ('@fold', str(fold)),
                            ('@sample', str(sample)), ('oml:name', measure),
                            ('oml:value', str(value))])
                        description['oml:run']['oml:output_data'][
                            'oml:evaluation'].append(current)
    return description
