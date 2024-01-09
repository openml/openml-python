# License: BSD 3-Clause
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator
from typing_extensions import Self

import arff
import xmltodict

PREFIX = "parameter_"
REQUIRED_ATTRIBUTES = [
    "repeat",
    "fold",
    "iteration",
    "evaluation",
    "selected",
]


@dataclass
class OpenMLTraceIteration:
    """
    OpenML Trace Iteration: parsed output from Run Trace call
    Exactly one of `setup_string` or `parameters` must be provided.

    Parameters
    ----------
    repeat : int
        repeat number (in case of no repeats: 0)

    fold : int
        fold number (in case of no folds: 0)

    iteration : int
        iteration number of optimization procedure

    setup_string : str, optional
        json string representing the parameters
        If not provided, ``parameters`` should be set.

    evaluation : double
        The evaluation that was awarded to this trace iteration.
        Measure is defined by the task

    selected : bool
        Whether this was the best of all iterations, and hence
        selected for making predictions. Per fold/repeat there
        should be only one iteration selected

    parameters : OrderedDict, optional
        Dictionary specifying parameter names and their values.
        If not provided, ``setup_string`` should be set.
    """

    repeat: int
    fold: int
    iteration: int

    evaluation: float
    selected: bool

    setup_string: dict[str, str] | None = None
    parameters: dict[str, str | int | float] | None = None

    def __post_init__(self) -> None:
        # TODO: refactor into one argument of type <str | OrderedDict>
        if self.setup_string and self.parameters:
            raise ValueError(
                "Can only be instantiated with either `setup_string` or `parameters` argument.",
            )

        if not (self.setup_string or self.parameters):
            raise ValueError(
                "Either `setup_string` or `parameters` needs to be passed as argument.",
            )

        if self.parameters is not None and not isinstance(self.parameters, dict):
            raise TypeError(
                "argument parameters is not an instance of OrderedDict, but %s"
                % str(type(self.parameters)),
            )

    def get_parameters(self) -> dict[str, Any]:
        """Get the parameters of this trace iteration."""
        # parameters have prefix 'parameter_'
        if self.setup_string:
            return {
                param[len(PREFIX) :]: json.loads(value)
                for param, value in self.setup_string.items()
            }

        assert self.parameters is not None
        return {param[len(PREFIX) :]: value for param, value in self.parameters.items()}


class OpenMLRunTrace:
    """OpenML Run Trace: parsed output from Run Trace call

    Parameters
    ----------
    run_id : int
        OpenML run id.

    trace_iterations : dict
        Mapping from key ``(repeat, fold, iteration)`` to an object of
        OpenMLTraceIteration.

    """

    def __init__(
        self,
        run_id: int | None,
        trace_iterations: dict[tuple[int, int, int], OpenMLTraceIteration],
    ):
        """Object to hold the trace content of a run.

        Parameters
        ----------
        run_id : int
            Id for which the trace content is to be stored.
        trace_iterations : List[List]
            The trace content obtained by running a flow on a task.
        """
        self.run_id = run_id
        self.trace_iterations = trace_iterations

    def get_selected_iteration(self, fold: int, repeat: int) -> int:
        """
        Returns the trace iteration that was marked as selected. In
        case multiple are marked as selected (should not happen) the
        first of these is returned

        Parameters
        ----------
        fold: int

        repeat: int

        Returns
        -------
        int
            The trace iteration from the given fold and repeat that was
            selected as the best iteration by the search procedure
        """
        for r, f, i in self.trace_iterations:
            if r == repeat and f == fold and self.trace_iterations[(r, f, i)].selected is True:
                return i
        raise ValueError(
            "Could not find the selected iteration for rep/fold %d/%d" % (repeat, fold),
        )

    @classmethod
    def generate(
        cls,
        attributes: list[tuple[str, str]],
        content: list[list[int | float | str]],
    ) -> OpenMLRunTrace:
        """Generates an OpenMLRunTrace.

        Generates the trace object from the attributes and content extracted
        while running the underlying flow.

        Parameters
        ----------
        attributes : list
            List of tuples describing the arff attributes.

        content : list
            List of lists containing information about the individual tuning
            runs.

        Returns
        -------
        OpenMLRunTrace
        """
        if content is None:
            raise ValueError("Trace content not available.")
        if attributes is None:
            raise ValueError("Trace attributes not available.")
        if len(content) == 0:
            raise ValueError("Trace content is empty.")
        if len(attributes) != len(content[0]):
            raise ValueError(
                "Trace_attributes and trace_content not compatible:"
                f" {attributes} vs {content[0]}",
            )

        return cls._trace_from_arff_struct(
            attributes=attributes,
            content=content,
            error_message="setup_string not allowed when constructing a "
            "trace object from run results.",
        )

    @classmethod
    def _from_filesystem(cls, file_path: str | Path) -> OpenMLRunTrace:
        """
        Logic to deserialize the trace from the filesystem.

        Parameters
        ----------
        file_path: str | Path
            File path where the trace arff is stored.

        Returns
        -------
        OpenMLRunTrace
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError("Trace file doesn't exist")

        with file_path.open("r") as fp:
            trace_arff = arff.load(fp)

        for trace_idx in range(len(trace_arff["data"])):
            # iterate over first three entrees of a trace row
            # (fold, repeat, trace_iteration) these should be int
            for line_idx in range(3):
                trace_arff["data"][trace_idx][line_idx] = int(
                    trace_arff["data"][trace_idx][line_idx],
                )

        return cls.trace_from_arff(trace_arff)

    def _to_filesystem(self, file_path: str | Path) -> None:
        """Serialize the trace object to the filesystem.

        Serialize the trace object as an arff.

        Parameters
        ----------
        file_path: str | Path
            File path where the trace arff will be stored.
        """
        trace_path = Path(file_path) / "trace.arff"

        trace_arff = arff.dumps(self.trace_to_arff())
        with trace_path.open("w") as f:
            f.write(trace_arff)

    def trace_to_arff(self) -> dict[str, Any]:
        """Generate the arff dictionary for uploading predictions to the server.

        Uses the trace object to generate an arff dictionary representation.

        Returns
        -------
        arff_dict : dict
            Dictionary representation of the ARFF file that will be uploaded.
            Contains information about the optimization trace.
        """
        if self.trace_iterations is None:
            raise ValueError("trace_iterations missing from the trace object")

        # attributes that will be in trace arff
        trace_attributes = [
            ("repeat", "NUMERIC"),
            ("fold", "NUMERIC"),
            ("iteration", "NUMERIC"),
            ("evaluation", "NUMERIC"),
            ("selected", ["true", "false"]),
        ]
        trace_attributes.extend(
            [
                (PREFIX + parameter, "STRING")
                for parameter in next(iter(self.trace_iterations.values())).get_parameters()
            ],
        )

        arff_dict: dict[str, Any] = {}
        data = []
        for trace_iteration in self.trace_iterations.values():
            tmp_list = []
            for _attr, _ in trace_attributes:
                if _attr.startswith(PREFIX):
                    attr = _attr[len(PREFIX) :]
                    value = trace_iteration.get_parameters()[attr]
                else:
                    attr = _attr
                    value = getattr(trace_iteration, attr)

                if attr == "selected":
                    tmp_list.append("true" if value else "false")
                else:
                    tmp_list.append(value)
            data.append(tmp_list)

        arff_dict["attributes"] = trace_attributes
        arff_dict["data"] = data
        # TODO allow to pass a trace description when running a flow
        arff_dict["relation"] = "Trace"
        return arff_dict

    @classmethod
    def trace_from_arff(cls, arff_obj: dict[str, Any]) -> OpenMLRunTrace:
        """Generate trace from arff trace.

        Creates a trace file from arff object (for example, generated by a
        local run).

        Parameters
        ----------
        arff_obj : dict
            LIAC arff obj, dict containing attributes, relation, data.

        Returns
        -------
        OpenMLRunTrace
        """
        attributes = arff_obj["attributes"]
        content = arff_obj["data"]
        return cls._trace_from_arff_struct(
            attributes=attributes,
            content=content,
            error_message="setup_string not supported for arff serialization",
        )

    @classmethod
    def _trace_from_arff_struct(
        cls,
        attributes: list[tuple[str, str]],
        content: list[list[int | float | str]],
        error_message: str,
    ) -> Self:
        """Generate a trace dictionary from ARFF structure.

        Parameters
        ----------
        cls : type
            The trace object to be created.
        attributes : list[tuple[str, str]]
            Attribute descriptions.
        content : list[list[int | float | str]]]
            List of instances.
        error_message : str
            Error message to raise if `setup_string` is in `attributes`.

        Returns
        -------
        OrderedDict
            A dictionary representing the trace.
        """
        trace = OrderedDict()
        attribute_idx = {att[0]: idx for idx, att in enumerate(attributes)}

        for required_attribute in REQUIRED_ATTRIBUTES:
            if required_attribute not in attribute_idx:
                raise ValueError("arff misses required attribute: %s" % required_attribute)
        if "setup_string" in attribute_idx:
            raise ValueError(error_message)

        # note that the required attributes can not be duplicated because
        # they are not parameters
        parameter_attributes = []
        for attribute in attribute_idx:
            if attribute in REQUIRED_ATTRIBUTES or attribute == "setup_string":
                continue

            if not attribute.startswith(PREFIX):
                raise ValueError(
                    f"Encountered unknown attribute {attribute} that does not start "
                    f"with prefix {PREFIX}",
                )

            parameter_attributes.append(attribute)

        for itt in content:
            repeat = int(itt[attribute_idx["repeat"]])
            fold = int(itt[attribute_idx["fold"]])
            iteration = int(itt[attribute_idx["iteration"]])
            evaluation = float(itt[attribute_idx["evaluation"]])
            selected_value = itt[attribute_idx["selected"]]
            if selected_value == "true":
                selected = True
            elif selected_value == "false":
                selected = False
            else:
                raise ValueError(
                    'expected {"true", "false"} value for selected field, '
                    "received: %s" % selected_value,
                )

            parameters = {
                attribute: itt[attribute_idx[attribute]] for attribute in parameter_attributes
            }

            current = OpenMLTraceIteration(
                repeat=repeat,
                fold=fold,
                iteration=iteration,
                setup_string=None,
                evaluation=evaluation,
                selected=selected,
                parameters=parameters,
            )
            trace[(repeat, fold, iteration)] = current

        return cls(None, trace)

    @classmethod
    def trace_from_xml(cls, xml: str | Path | IO) -> OpenMLRunTrace:
        """Generate trace from xml.

        Creates a trace file from the xml description.

        Parameters
        ----------
        xml : string | file-like object
            An xml description that can be either a `string` or a file-like
            object.

        Returns
        -------
        run : OpenMLRunTrace
            Object containing the run id and a dict containing the trace
            iterations.
        """
        if isinstance(xml, Path):
            xml = str(xml.absolute())

        result_dict = xmltodict.parse(xml, force_list=("oml:trace_iteration",))["oml:trace"]

        run_id = result_dict["oml:run_id"]
        trace = OrderedDict()

        if "oml:trace_iteration" not in result_dict:
            raise ValueError("Run does not contain valid trace. ")
        if not isinstance(result_dict["oml:trace_iteration"], list):
            raise TypeError(type(result_dict["oml:trace_iteration"]))

        for itt in result_dict["oml:trace_iteration"]:
            repeat = int(itt["oml:repeat"])
            fold = int(itt["oml:fold"])
            iteration = int(itt["oml:iteration"])
            setup_string = json.loads(itt["oml:setup_string"])
            evaluation = float(itt["oml:evaluation"])
            selected_value = itt["oml:selected"]
            if selected_value == "true":
                selected = True
            elif selected_value == "false":
                selected = False
            else:
                raise ValueError(
                    'expected {"true", "false"} value for '
                    "selected field, received: %s" % selected_value,
                )

            current = OpenMLTraceIteration(
                repeat=repeat,
                fold=fold,
                iteration=iteration,
                setup_string=setup_string,
                evaluation=evaluation,
                selected=selected,
            )
            trace[(repeat, fold, iteration)] = current

        return cls(run_id, trace)

    @classmethod
    def merge_traces(cls, traces: list[OpenMLRunTrace]) -> OpenMLRunTrace:
        """Merge multiple traces into a single trace.

        Parameters
        ----------
        cls : type
            Type of the trace object to be created.
        traces : List[OpenMLRunTrace]
            List of traces to merge.

        Returns
        -------
        OpenMLRunTrace
            A trace object representing the merged traces.

        Raises
        ------
        ValueError
            If the parameters in the iterations of the traces being merged are not equal.
            If a key (repeat, fold, iteration) is encountered twice while merging the traces.
        """
        merged_trace: dict[tuple[int, int, int], OpenMLTraceIteration] = {}

        previous_iteration = None
        for trace in traces:
            for iteration in trace:
                key = (iteration.repeat, iteration.fold, iteration.iteration)

                assert iteration.parameters is not None
                param_keys = iteration.parameters.keys()

                if previous_iteration is not None:
                    trace_itr = merged_trace[previous_iteration]

                    assert trace_itr.parameters is not None
                    trace_itr_keys = trace_itr.parameters.keys()

                    if list(param_keys) != list(trace_itr_keys):
                        raise ValueError(
                            "Cannot merge traces because the parameters are not equal: "
                            "{} vs {}".format(
                                list(trace_itr.parameters.keys()),
                                list(iteration.parameters.keys()),
                            ),
                        )

                if key in merged_trace:
                    raise ValueError(
                        f"Cannot merge traces because key '{key}' was encountered twice",
                    )

                merged_trace[key] = iteration
                previous_iteration = key

        return cls(None, merged_trace)

    def __repr__(self) -> str:
        return "[Run id: {}, {} trace iterations]".format(
            -1 if self.run_id is None else self.run_id,
            len(self.trace_iterations),
        )

    def __iter__(self) -> Iterator[OpenMLTraceIteration]:
        yield from self.trace_iterations.values()
