# License: BSD 3-Clause
from __future__ import annotations

import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Sequence,
)

import arff
import numpy as np
import pandas as pd

import openml
import openml._api_calls
from openml.base import OpenMLBase
from openml.exceptions import PyOpenMLError
from openml.flows import OpenMLFlow, get_flow
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLTask,
    TaskType,
    get_task,
)

if TYPE_CHECKING:
    from openml.runs.trace import OpenMLRunTrace


class OpenMLRun(OpenMLBase):
    """OpenML Run: result of running a model on an OpenML dataset.

    Parameters
    ----------
    task_id: int
        The ID of the OpenML task associated with the run.
    flow_id: int
        The ID of the OpenML flow associated with the run.
    dataset_id: int
        The ID of the OpenML dataset used for the run.
    setup_string: str
        The setup string of the run.
    output_files: Dict[str, int]
        Specifies where each related file can be found.
    setup_id: int
        An integer representing the ID of the setup used for the run.
    tags: List[str]
        Representing the tags associated with the run.
    uploader: int
        User ID of the uploader.
    uploader_name: str
        The name of the person who uploaded the run.
    evaluations: Dict
        Representing the evaluations of the run.
    fold_evaluations: Dict
        The evaluations of the run for each fold.
    sample_evaluations: Dict
        The evaluations of the run for each sample.
    data_content: List[List]
        The predictions generated from executing this run.
    trace: OpenMLRunTrace
        The trace containing information on internal model evaluations of this run.
    model: object
        The untrained model that was evaluated in the run.
    task_type: str
        The type of the OpenML task associated with the run.
    task_evaluation_measure: str
        The evaluation measure used for the task.
    flow_name: str
        The name of the OpenML flow associated with the run.
    parameter_settings: list[OrderedDict]
        Representing the parameter settings used for the run.
    predictions_url: str
        The URL of the predictions file.
    task: OpenMLTask
        An instance of the OpenMLTask class, representing the OpenML task associated
        with the run.
    flow: OpenMLFlow
        An instance of the OpenMLFlow class, representing the OpenML flow associated
        with the run.
    run_id: int
        The ID of the run.
    description_text: str, optional
        Description text to add to the predictions file. If left None, is set to the
        time the arff file is generated.
    run_details: str, optional (default=None)
        Description of the run stored in the run meta-data.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_id: int,
        flow_id: int | None,
        dataset_id: int | None,
        setup_string: str | None = None,
        output_files: dict[str, int] | None = None,
        setup_id: int | None = None,
        tags: list[str] | None = None,
        uploader: int | None = None,
        uploader_name: str | None = None,
        evaluations: dict | None = None,
        fold_evaluations: dict | None = None,
        sample_evaluations: dict | None = None,
        data_content: list[list] | None = None,
        trace: OpenMLRunTrace | None = None,
        model: object | None = None,
        task_type: str | None = None,
        task_evaluation_measure: str | None = None,
        flow_name: str | None = None,
        parameter_settings: list[dict[str, Any]] | None = None,
        predictions_url: str | None = None,
        task: OpenMLTask | None = None,
        flow: OpenMLFlow | None = None,
        run_id: int | None = None,
        description_text: str | None = None,
        run_details: str | None = None,
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
        self.run_details = run_details
        self._predictions = None

    @property
    def predictions(self) -> pd.DataFrame:
        """Return a DataFrame with predictions for this run"""
        if self._predictions is None:
            if self.data_content:
                arff_dict = self._generate_arff_dict()
            elif self.predictions_url:
                arff_text = openml._api_calls._download_text_file(self.predictions_url)
                arff_dict = arff.loads(arff_text)
            else:
                raise RuntimeError("Run has no predictions.")
            self._predictions = pd.DataFrame(
                arff_dict["data"],
                columns=[name for name, _ in arff_dict["attributes"]],
            )
        return self._predictions

    @property
    def id(self) -> int | None:
        """The ID of the run, None if not uploaded to the server yet."""
        return self.run_id

    def _evaluation_summary(self, metric: str) -> str:
        """Summarizes the evaluation of a metric over all folds.

        The fold scores for the metric must exist already. During run creation,
        by default, the MAE for OpenMLRegressionTask and the accuracy for
        OpenMLClassificationTask/OpenMLLearningCurveTasktasks are computed.

        If repetition exist, we take the mean over all repetitions.

        Parameters
        ----------
        metric: str
            Name of an evaluation metric that was used to compute fold scores.

        Returns
        -------
        metric_summary: str
            A formatted string that displays the metric's evaluation summary.
            The summary consists of the mean and std.
        """
        if self.fold_evaluations is None:
            raise ValueError("No fold evaluations available.")
        fold_score_lists = self.fold_evaluations[metric].values()

        # Get the mean and std over all repetitions
        rep_means = [np.mean(list(x.values())) for x in fold_score_lists]
        rep_stds = [np.std(list(x.values())) for x in fold_score_lists]

        return f"{np.mean(rep_means):.4f} +- {np.mean(rep_stds):.4f}"

    def _get_repr_body_fields(self) -> Sequence[tuple[str, str | int | list[str]]]:
        """Collect all information to display in the __repr__ body."""
        # Set up fields
        fields = {
            "Uploader Name": self.uploader_name,
            "Metric": self.task_evaluation_measure,
            "Run ID": self.run_id,
            "Task ID": self.task_id,
            "Task Type": self.task_type,
            "Task URL": openml.tasks.OpenMLTask.url_for_id(self.task_id),
            "Flow ID": self.flow_id,
            "Flow Name": self.flow_name,
            "Flow URL": (
                openml.flows.OpenMLFlow.url_for_id(self.flow_id)
                if self.flow_id is not None
                else None
            ),
            "Setup ID": self.setup_id,
            "Setup String": self.setup_string,
            "Dataset ID": self.dataset_id,
            "Dataset URL": (
                openml.datasets.OpenMLDataset.url_for_id(self.dataset_id)
                if self.dataset_id is not None
                else None
            ),
        }

        # determines the order of the initial fields in which the information will be printed
        order = ["Uploader Name", "Uploader Profile", "Metric", "Result"]

        if self.uploader is not None:
            fields["Uploader Profile"] = f"{openml.config.get_server_base_url()}/u/{self.uploader}"
        if self.run_id is not None:
            fields["Run URL"] = self.openml_url
        if self.evaluations is not None and self.task_evaluation_measure in self.evaluations:
            fields["Result"] = self.evaluations[self.task_evaluation_measure]
        elif self.fold_evaluations is not None:
            # -- Add locally computed summary values if possible
            if "predictive_accuracy" in self.fold_evaluations:
                # OpenMLClassificationTask; OpenMLLearningCurveTask
                result_field = "Local Result - Accuracy (+- STD)"
                fields[result_field] = self._evaluation_summary("predictive_accuracy")
                order.append(result_field)
            elif "mean_absolute_error" in self.fold_evaluations:
                # OpenMLRegressionTask
                result_field = "Local Result - MAE (+- STD)"
                fields[result_field] = self._evaluation_summary("mean_absolute_error")
                order.append(result_field)

            if "usercpu_time_millis" in self.fold_evaluations:
                # Runtime should be available for most tasks types
                rt_field = "Local Runtime - ms (+- STD)"
                fields[rt_field] = self._evaluation_summary("usercpu_time_millis")
                order.append(rt_field)

        # determines the remaining order
        order += [
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
        return [
            (key, "None" if fields[key] is None else fields[key])  # type: ignore
            for key in order
            if key in fields
        ]

    @classmethod
    def from_filesystem(cls, directory: str | Path, expect_model: bool = True) -> OpenMLRun:  # noqa: FBT001, FBT002
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

        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError("Could not find folder")

        description_path = directory / "description.xml"
        predictions_path = directory / "predictions.arff"
        trace_path = directory / "trace.arff"
        model_path = directory / "model.pkl"

        if not description_path.is_file():
            raise ValueError("Could not find description.xml")
        if not predictions_path.is_file():
            raise ValueError("Could not find predictions.arff")
        if (not model_path.is_file()) and expect_model:
            raise ValueError("Could not find model.pkl")

        with description_path.open() as fht:
            xml_string = fht.read()
        run = openml.runs.functions._create_run_from_xml(xml_string, from_server=False)

        if run.flow_id is None:
            flow = openml.flows.OpenMLFlow.from_filesystem(directory)
            run.flow = flow
            run.flow_name = flow.name

        with predictions_path.open() as fht:
            predictions = arff.load(fht)
            run.data_content = predictions["data"]

        if model_path.is_file():
            # note that it will load the model if the file exists, even if
            # expect_model is False
            with model_path.open("rb") as fhb:
                run.model = pickle.load(fhb)  # noqa: S301

        if trace_path.is_file():
            run.trace = openml.runs.OpenMLRunTrace._from_filesystem(trace_path)

        return run

    def to_filesystem(
        self,
        directory: str | Path,
        store_model: bool = True,  # noqa: FBT001, FBT002
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
            raise ValueError("Run should have been executed (and contain " "model / predictions)")
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        if any(directory.iterdir()):
            raise ValueError(f"Output directory {directory.expanduser().resolve()} should be empty")

        run_xml = self._to_xml()
        predictions_arff = arff.dumps(self._generate_arff_dict())

        # It seems like typing does not allow to define the same variable multiple times
        with (directory / "description.xml").open("w") as fh:
            fh.write(run_xml)
        with (directory / "predictions.arff").open("w") as fh:
            fh.write(predictions_arff)
        if store_model:
            with (directory / "model.pkl").open("wb") as fh_b:
                pickle.dump(self.model, fh_b)

        if self.flow_id is None and self.flow is not None:
            self.flow.to_filesystem(directory)

        if self.trace is not None:
            self.trace._to_filesystem(directory)

    def _generate_arff_dict(self) -> OrderedDict[str, Any]:
        """Generates the arff dictionary for uploading predictions to the
        server.

        Assumes that the run has been executed.

        The order of the attributes follows the order defined by the Client API for R.

        Returns
        -------
        arf_dict : dict
            Dictionary representation of the ARFF file that will be uploaded.
            Contains predictions and information about the run environment.
        """
        if self.data_content is None:
            raise ValueError("Run has not been executed.")
        if self.flow is None:
            assert self.flow_id is not None, "Run has no associated flow id!"
            self.flow = get_flow(self.flow_id)

        if self.description_text is None:
            self.description_text = time.strftime("%c")
        task = get_task(self.task_id)

        arff_dict = OrderedDict()  # type: 'OrderedDict[str, Any]'
        arff_dict["data"] = self.data_content
        arff_dict["description"] = self.description_text
        arff_dict["relation"] = f"openml_task_{task.task_id}_predictions"

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
                    + [("prediction", class_labels), ("correct", class_labels)]
                    + [
                        ("confidence." + class_labels[i], "NUMERIC")
                        for i in range(len(class_labels))
                    ]
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
                    arff_dict["attributes"] + prediction_and_true + prediction_confidences
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
            raise NotImplementedError(f"Task type {task.task_type!s} is not yet supported.")

        return arff_dict

    def get_metric_fn(self, sklearn_fn: Callable, kwargs: dict | None = None) -> np.ndarray:  # noqa: PLR0915, PLR0912, C901
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
        kwargs : dict
            kwargs for the function

        Returns
        -------
        scores : ndarray of scores of length num_folds * num_repeats
            metric results
        """
        kwargs = kwargs if kwargs else {}
        if self.data_content is not None and self.task_id is not None:
            predictions_arff = self._generate_arff_dict()
        elif (self.output_files is not None) and ("predictions" in self.output_files):
            predictions_file_url = openml._api_calls._file_id_to_url(
                self.output_files["predictions"],
                "predictions.arff",
            )
            response = openml._api_calls._download_text_file(predictions_file_url)
            predictions_arff = arff.loads(response)
            # TODO: make this a stream reader
        else:
            raise ValueError(
                "Run should have been locally executed or " "contain outputfile reference.",
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

        def _attribute_list_to_dict(attribute_list):  # type: ignore
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

        if task.task_type_id in (TaskType.SUPERVISED_CLASSIFICATION, TaskType.LEARNING_CURVE):
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
                "Predicted and Correct do not have equal values:" f" {pred!s} Vs. {corr!s}",
            )

        # TODO: these could be cached
        values_predict: dict[int, dict[int, dict[int, list[float]]]] = {}
        values_correct: dict[int, dict[int, dict[int, list[float]]]] = {}
        for _line_idx, line in enumerate(predictions_arff["data"]):
            rep = line[repeat_idx]
            fold = line[fold_idx]
            samp = line[sample_idx] if has_samples else 0

            if task.task_type_id in [
                TaskType.SUPERVISED_CLASSIFICATION,
                TaskType.LEARNING_CURVE,
            ]:
                prediction = predictions_arff["attributes"][predicted_idx][1].index(
                    line[predicted_idx],
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
        for rep in values_predict:
            for fold in values_predict[rep]:
                last_sample = len(values_predict[rep][fold]) - 1
                y_pred = values_predict[rep][fold][last_sample]
                y_true = values_correct[rep][fold][last_sample]
                scores.append(sklearn_fn(y_true, y_pred, **kwargs))
        return np.array(scores)

    def _parse_publish_response(self, xml_response: dict) -> None:
        """Parse the id from the xml_response and assign it to self."""
        self.run_id = int(xml_response["oml:upload_run"]["oml:run_id"])

    def _get_file_elements(self) -> dict:
        """Get file_elements to upload to the server.

        Derived child classes should overwrite this method as necessary.
        The description field will be populated automatically if not provided.
        """
        if self.parameter_settings is None and self.model is None:
            raise PyOpenMLError(
                "OpenMLRun must contain a model or be initialized with parameter_settings.",
            )
        if self.flow_id is None:
            if self.flow is None:
                raise PyOpenMLError(
                    "OpenMLRun object does not contain a flow id or reference to OpenMLFlow "
                    "(these should have been added while executing the task). ",
                )

            # publish the linked Flow before publishing the run.
            self.flow.publish()
            self.flow_id = self.flow.flow_id

        if self.parameter_settings is None:
            if self.flow is None:
                assert self.flow_id is not None  # for mypy
                self.flow = openml.flows.get_flow(self.flow_id)
            self.parameter_settings = self.flow.extension.obtain_parameter_values(
                self.flow,
                self.model,
            )

        file_elements = {"description": ("description.xml", self._to_xml())}

        if self.error_message is None:
            predictions = arff.dumps(self._generate_arff_dict())
            file_elements["predictions"] = ("predictions.arff", predictions)

        if self.trace is not None:
            trace_arff = arff.dumps(self.trace.trace_to_arff())
            file_elements["trace"] = ("trace.arff", trace_arff)
        return file_elements

    def _to_dict(self) -> dict[str, dict]:  # noqa: PLR0912, C901
        """Creates a dictionary representation of self."""
        description = OrderedDict()  # type: 'OrderedDict'
        description["oml:run"] = OrderedDict()
        description["oml:run"]["@xmlns:oml"] = "http://openml.org/openml"
        description["oml:run"]["oml:task_id"] = self.task_id
        description["oml:run"]["oml:flow_id"] = self.flow_id
        if self.setup_string is not None:
            description["oml:run"]["oml:setup_string"] = self.setup_string
        if self.error_message is not None:
            description["oml:run"]["oml:error_message"] = self.error_message
        if self.run_details is not None:
            description["oml:run"]["oml:run_details"] = self.run_details
        description["oml:run"]["oml:parameter_setting"] = self.parameter_settings
        if self.tags is not None:
            description["oml:run"]["oml:tag"] = self.tags
        if (self.fold_evaluations is not None and len(self.fold_evaluations) > 0) or (
            self.sample_evaluations is not None and len(self.sample_evaluations) > 0
        ):
            description["oml:run"]["oml:output_data"] = OrderedDict()
            description["oml:run"]["oml:output_data"]["oml:evaluation"] = []
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
                            ],
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
                                ],
                            )
                            description["oml:run"]["oml:output_data"]["oml:evaluation"].append(
                                current,
                            )
        return description
