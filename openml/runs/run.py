# License: BSD 3-Clause

from collections import OrderedDict
import pickle
import time
from typing import Any, IO, TextIO, List, Union, Tuple, Optional, Dict  # noqa F401
import os

import arff
import numpy as np

import openml
import openml._api_calls
from openml.base import OpenMLBase
from ..exceptions import PyOpenMLError
from ..flows import get_flow
from ..tasks import (
    get_task,
    TaskType,
    OpenMLClassificationTask,
    OpenMLLearningCurveTask,
    OpenMLClusteringTask,
    OpenMLRegressionTask,
)


class OpenMLRun(OpenMLBase):
    """OpenML Run: result of running a model on an openml dataset.

    Parameters
    ----------
    task_id: int
    flow_id: int
    dataset_id: int
    setup_string: str
    output_files: Dict[str, str]
        A dictionary that specifies where each related file can be found.
    setup_id: int
    tags: List[str]
    uploader: int
        User ID of the uploader.
    uploader_name: str
    evaluations: Dict
    fold_evaluations: Dict
    sample_evaluations: Dict
    data_content: List[List]
        The predictions generated from executing this run.
    trace: OpenMLRunTrace
    model: object
    task_type: str
    task_evaluation_measure: str
    flow_name: str
    parameter_settings: List[OrderedDict]
    predictions_url: str
    task: OpenMLTask
    flow: OpenMLFlow
    run_id: int
    description_text: str, optional
        Description text to add to the predictions file.
        If left None,
    """

    def __init__(
        self,
        task_id,
        flow_id,
        dataset_id,
        setup_string=None,
        output_files=None,
        setup_id=None,
        tags=None,
        uploader=None,
        uploader_name=None,
        evaluations=None,
        fold_evaluations=None,
        sample_evaluations=None,
        data_content=None,
        trace=None,
        model=None,
        task_type=None,
        task_evaluation_measure=None,
        flow_name=None,
        parameter_settings=None,
        predictions_url=None,
        task=None,
        flow=None,
        run_id=None,
        description_text=None,
    ):
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
        self.description_text = description_text

    @property
    def id(self) -> Optional[int]:
        return self.run_id

    def _get_repr_body_fields(self) -> List[Tuple[str, Union[str, int, List[str]]]]:
        """ Collect all information to display in the __repr__ body. """
        fields = {
            "Uploader Name": self.uploader_name,
            "Metric": self.task_evaluation_measure,
            "Run ID": self.run_id,
            "Task ID": self.task_id,
            "Task Type": self.task_type,
            "Task URL": openml.tasks.OpenMLTask.url_for_id(self.task_id),
            "Flow ID": self.flow_id,
            "Flow Name": self.flow_name,
            "Flow URL": openml.flows.OpenMLFlow.url_for_id(self.flow_id),
            "Setup ID": self.setup_id,
            "Setup String": self.setup_string,
            "Dataset ID": self.dataset_id,
            "Dataset URL": openml.datasets.OpenMLDataset.url_for_id(self.dataset_id),
        }
        if self.uploader is not None:
            fields["Uploader Profile"] = "{}/u/{}".format(
                openml.config.get_server_base_url(), self.uploader
            )
        if self.run_id is not None:
            fields["Run URL"] = self.openml_url
        if self.evaluations is not None and self.task_evaluation_measure in self.evaluations:
            fields["Result"] = self.evaluations[self.task_evaluation_measure]

        # determines the order in which the information will be printed
        order = [
            "Uploader Name",
            "Uploader Profile",
            "Metric",
            "Result",
            "Run ID",
            "Run URL",
            "Task ID",
            "Task Type",
            "Task URL",
            "Flow ID",
            "Flow Name",
            "Flow URL",
            "Setup ID",
            "Setup String",
            "Dataset ID",
            "Dataset URL",
        ]
        return [(key, fields[key]) for key in order if key in fields]

    @classmethod
    def from_filesystem(cls, directory: str, expect_model: bool = True) -> "OpenMLRun":
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
            raise ValueError("Could not find folder")

        description_path = os.path.join(directory, "description.xml")
        predictions_path = os.path.join(directory, "predictions.arff")
        trace_path = os.path.join(directory, "trace.arff")
        model_path = os.path.join(directory, "model.pkl")

        if not os.path.isfile(description_path):
            raise ValueError("Could not find description.xml")
        if not os.path.isfile(predictions_path):
            raise ValueError("Could not find predictions.arff")
        if not os.path.isfile(model_path) and expect_model:
            raise ValueError("Could not find model.pkl")

        with open(description_path, "r") as fht:
            xml_string = fht.read()
        run = openml.runs.functions._create_run_from_xml(xml_string, from_server=False)

        if run.flow_id is None:
            flow = openml.flows.OpenMLFlow.from_filesystem(directory)
            run.flow = flow
            run.flow_name = flow.name

        with open(predictions_path, "r") as fht:
            predictions = arff.load(fht)
            run.data_content = predictions["data"]

        if os.path.isfile(model_path):
            # note that it will load the model if the file exists, even if
            # expect_model is False
            with open(model_path, "rb") as fhb:
                run.model = pickle.load(fhb)

        if os.path.isfile(trace_path):
            run.trace = openml.runs.OpenMLRunTrace._from_filesystem(trace_path)

        return run

    def to_filesystem(self, directory: str, store_model: bool = True,) -> None:
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
            raise ValueError("Run should have been executed (and contain " "model / predictions)")

        os.makedirs(directory, exist_ok=True)
        if not os.listdir(directory) == []:
            raise ValueError(
                "Output directory {} should be empty".format(os.path.abspath(directory))
            )

        run_xml = self._to_xml()
        predictions_arff = arff.dumps(self._generate_arff_dict())

        # It seems like typing does not allow to define the same variable multiple times
        with open(os.path.join(directory, "description.xml"), "w") as fh:  # type: TextIO
            fh.write(run_xml)
        with open(os.path.join(directory, "predictions.arff"), "w") as fh:
            fh.write(predictions_arff)
        if store_model:
            with open(os.path.join(directory, "model.pkl"), "wb") as fh_b:  # type: IO[bytes]
                pickle.dump(self.model, fh_b)

        if self.flow_id is None:
            self.flow.to_filesystem(directory)

        if self.trace is not None:
            self.trace._to_filesystem(directory)

    def _generate_arff_dict(self) -> "OrderedDict[str, Any]":
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
            raise ValueError("Run has not been executed.")
        if self.flow is None:
            self.flow = get_flow(self.flow_id)

        if self.description_text is None:
            self.description_text = time.strftime("%c")
        task = get_task(self.task_id)

        arff_dict = OrderedDict()  # type: 'OrderedDict[str, Any]'
        arff_dict["data"] = self.data_content
        arff_dict["description"] = self.description_text
        arff_dict["relation"] = "openml_task_{}_predictions".format(task.task_id)

        if isinstance(task, OpenMLLearningCurveTask):
            class_labels = task.class_labels
            instance_specifications = [
                ("repeat", "NUMERIC"),
                ("fold", "NUMERIC"),
                ("sample", "NUMERIC"),
                ("row_id", "NUMERIC"),
            ]

            arff_dict["attributes"] = instance_specifications
            if class_labels is not None:
                arff_dict["attributes"] = (
                    arff_dict["attributes"]
                    + [
                        ("confidence." + class_labels[i], "NUMERIC")
                        for i in range(len(class_labels))
                    ]
                    + [("prediction", class_labels), ("correct", class_labels)]
                )
            else:
                raise ValueError("The task has no class labels")

        elif isinstance(task, OpenMLClassificationTask):
            class_labels = task.class_labels
            instance_specifications = [
                ("repeat", "NUMERIC"),
                ("fold", "NUMERIC"),
                ("sample", "NUMERIC"),  # Legacy
                ("row_id", "NUMERIC"),
            ]

            arff_dict["attributes"] = instance_specifications
            if class_labels is not None:
                prediction_confidences = [
                    ("confidence." + class_labels[i], "NUMERIC") for i in range(len(class_labels))
                ]
                prediction_and_true = [("prediction", class_labels), ("correct", class_labels)]
                arff_dict["attributes"] = (
                    arff_dict["attributes"] + prediction_confidences + prediction_and_true
                )
            else:
                raise ValueError("The task has no class labels")

        elif isinstance(task, OpenMLRegressionTask):
            arff_dict["attributes"] = [
                ("repeat", "NUMERIC"),
                ("fold", "NUMERIC"),
                ("row_id", "NUMERIC"),
                ("prediction", "NUMERIC"),
                ("truth", "NUMERIC"),
            ]

        elif isinstance(task, OpenMLClusteringTask):
            arff_dict["attributes"] = [
                ("repeat", "NUMERIC"),
                ("fold", "NUMERIC"),
                ("row_id", "NUMERIC"),
                ("cluster", "NUMERIC"),
            ]

        else:
            raise NotImplementedError("Task type %s is not yet supported." % str(task.task_type))

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
        elif "predictions" in self.output_files:
            predictions_file_url = openml._api_calls._file_id_to_url(
                self.output_files["predictions"], "predictions.arff",
            )
            response = openml._api_calls._download_text_file(predictions_file_url)
            predictions_arff = arff.loads(response)
            # TODO: make this a stream reader
        else:
            raise ValueError(
                "Run should have been locally executed or " "contain outputfile reference."
            )

        # Need to know more about the task to compute scores correctly
        task = get_task(self.task_id)

        attribute_names = [att[0] for att in predictions_arff["attributes"]]
        if (
            task.task_type_id in [TaskType.SUPERVISED_CLASSIFICATION, TaskType.LEARNING_CURVE]
            and "correct" not in attribute_names
        ):
            raise ValueError('Attribute "correct" should be set for ' "classification task runs")
        if task.task_type_id == TaskType.SUPERVISED_REGRESSION and "truth" not in attribute_names:
            raise ValueError('Attribute "truth" should be set for ' "regression task runs")
        if task.task_type_id != TaskType.CLUSTERING and "prediction" not in attribute_names:
            raise ValueError('Attribute "predict" should be set for ' "supervised task runs")

        def _attribute_list_to_dict(attribute_list):
            # convenience function: Creates a mapping to map from the name of
            # attributes present in the arff prediction file to their index.
            # This is necessary because the number of classes can be different
            # for different tasks.
            res = OrderedDict()
            for idx in range(len(attribute_list)):
                res[attribute_list[idx][0]] = idx
            return res

        attribute_dict = _attribute_list_to_dict(predictions_arff["attributes"])

        repeat_idx = attribute_dict["repeat"]
        fold_idx = attribute_dict["fold"]
        predicted_idx = attribute_dict["prediction"]  # Assume supervised task

        if (
            task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION
            or task.task_type_id == TaskType.LEARNING_CURVE
        ):
            correct_idx = attribute_dict["correct"]
        elif task.task_type_id == TaskType.SUPERVISED_REGRESSION:
            correct_idx = attribute_dict["truth"]
        has_samples = False
        if "sample" in attribute_dict:
            sample_idx = attribute_dict["sample"]
            has_samples = True

        if (
            predictions_arff["attributes"][predicted_idx][1]
            != predictions_arff["attributes"][correct_idx][1]
        ):
            pred = predictions_arff["attributes"][predicted_idx][1]
            corr = predictions_arff["attributes"][correct_idx][1]
            raise ValueError(
                "Predicted and Correct do not have equal values:"
                " %s Vs. %s" % (str(pred), str(corr))
            )

        # TODO: these could be cached
        values_predict = {}
        values_correct = {}
        for line_idx, line in enumerate(predictions_arff["data"]):
            rep = line[repeat_idx]
            fold = line[fold_idx]
            if has_samples:
                samp = line[sample_idx]
            else:
                samp = 0  # No learning curve sample, always 0

            if task.task_type_id in [
                TaskType.SUPERVISED_CLASSIFICATION,
                TaskType.LEARNING_CURVE,
            ]:
                prediction = predictions_arff["attributes"][predicted_idx][1].index(
                    line[predicted_idx]
                )
                correct = predictions_arff["attributes"][predicted_idx][1].index(line[correct_idx])
            elif task.task_type_id == TaskType.SUPERVISED_REGRESSION:
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

    def _parse_publish_response(self, xml_response: Dict):
        """ Parse the id from the xml_response and assign it to self. """
        self.run_id = int(xml_response["oml:upload_run"]["oml:run_id"])

    def _get_file_elements(self) -> Dict:
        """ Get file_elements to upload to the server.

        Derived child classes should overwrite this method as necessary.
        The description field will be populated automatically if not provided.
        """
        if self.parameter_settings is None and self.model is None:
            raise PyOpenMLError(
                "OpenMLRun must contain a model or be initialized with parameter_settings."
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
                self.flow, self.model,
            )

        file_elements = {"description": ("description.xml", self._to_xml())}

        if self.error_message is None:
            predictions = arff.dumps(self._generate_arff_dict())
            file_elements["predictions"] = ("predictions.arff", predictions)

        if self.trace is not None:
            trace_arff = arff.dumps(self.trace.trace_to_arff())
            file_elements["trace"] = ("trace.arff", trace_arff)
        return file_elements

    def _to_dict(self) -> "OrderedDict[str, OrderedDict]":
        """ Creates a dictionary representation of self. """
        description = OrderedDict()  # type: 'OrderedDict'
        description["oml:run"] = OrderedDict()
        description["oml:run"]["@xmlns:oml"] = "http://openml.org/openml"
        description["oml:run"]["oml:task_id"] = self.task_id
        description["oml:run"]["oml:flow_id"] = self.flow_id
        if self.error_message is not None:
            description["oml:run"]["oml:error_message"] = self.error_message
        description["oml:run"]["oml:parameter_setting"] = self.parameter_settings
        if self.tags is not None:
            description["oml:run"]["oml:tag"] = self.tags  # Tags describing the run
        if (self.fold_evaluations is not None and len(self.fold_evaluations) > 0) or (
            self.sample_evaluations is not None and len(self.sample_evaluations) > 0
        ):
            description["oml:run"]["oml:output_data"] = OrderedDict()
            description["oml:run"]["oml:output_data"]["oml:evaluation"] = list()
        if self.fold_evaluations is not None:
            for measure in self.fold_evaluations:
                for repeat in self.fold_evaluations[measure]:
                    for fold, value in self.fold_evaluations[measure][repeat].items():
                        current = OrderedDict(
                            [
                                ("@repeat", str(repeat)),
                                ("@fold", str(fold)),
                                ("oml:name", measure),
                                ("oml:value", str(value)),
                            ]
                        )
                        description["oml:run"]["oml:output_data"]["oml:evaluation"].append(current)
        if self.sample_evaluations is not None:
            for measure in self.sample_evaluations:
                for repeat in self.sample_evaluations[measure]:
                    for fold in self.sample_evaluations[measure][repeat]:
                        for sample, value in self.sample_evaluations[measure][repeat][fold].items():
                            current = OrderedDict(
                                [
                                    ("@repeat", str(repeat)),
                                    ("@fold", str(fold)),
                                    ("@sample", str(sample)),
                                    ("oml:name", measure),
                                    ("oml:value", str(value)),
                                ]
                            )
                            description["oml:run"]["oml:output_data"]["oml:evaluation"].append(
                                current
                            )
        return description
