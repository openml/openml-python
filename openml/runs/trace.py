# License: BSD 3-Clause

from collections import OrderedDict
import json
import os
from typing import List, Tuple, Optional  # noqa F401

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


class OpenMLRunTrace(object):
    """OpenML Run Trace: parsed output from Run Trace call

    Parameters
    ----------
    run_id : int
        OpenML run id.

    trace_iterations : dict
        Mapping from key ``(repeat, fold, iteration)`` to an object of
        OpenMLTraceIteration.

    """

    def __init__(self, run_id, trace_iterations):
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
        ----------
        int
            The trace iteration from the given fold and repeat that was
            selected as the best iteration by the search procedure
        """
        for (r, f, i) in self.trace_iterations:
            if r == repeat and f == fold and self.trace_iterations[(r, f, i)].selected is True:
                return i
        raise ValueError(
            "Could not find the selected iteration for rep/fold %d/%d" % (repeat, fold)
        )

    @classmethod
    def generate(cls, attributes, content):
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
        elif attributes is None:
            raise ValueError("Trace attributes not available.")
        elif len(content) == 0:
            raise ValueError("Trace content is empty.")
        elif len(attributes) != len(content[0]):
            raise ValueError(
                "Trace_attributes and trace_content not compatible:"
                " %s vs %s" % (attributes, content[0])
            )

        return cls._trace_from_arff_struct(
            attributes=attributes,
            content=content,
            error_message="setup_string not allowed when constructing a "
            "trace object from run results.",
        )

    @classmethod
    def _from_filesystem(cls, file_path: str) -> "OpenMLRunTrace":
        """
        Logic to deserialize the trace from the filesystem.

        Parameters
        ----------
        file_path: str
            File path where the trace arff is stored.

        Returns
        ----------
        OpenMLRunTrace
        """
        if not os.path.isfile(file_path):
            raise ValueError("Trace file doesn't exist")

        with open(file_path, "r") as fp:
            trace_arff = arff.load(fp)

        for trace_idx in range(len(trace_arff["data"])):
            # iterate over first three entrees of a trace row
            # (fold, repeat, trace_iteration) these should be int
            for line_idx in range(3):
                trace_arff["data"][trace_idx][line_idx] = int(
                    trace_arff["data"][trace_idx][line_idx]
                )

        return cls.trace_from_arff(trace_arff)

    def _to_filesystem(self, file_path):
        """Serialize the trace object to the filesystem.

        Serialize the trace object as an arff.

        Parameters
        ----------
        file_path: str
            File path where the trace arff will be stored.
        """

        trace_arff = arff.dumps(self.trace_to_arff())
        with open(os.path.join(file_path, "trace.arff"), "w") as f:
            f.write(trace_arff)

    def trace_to_arff(self):
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
            ]
        )

        arff_dict = OrderedDict()
        data = []
        for trace_iteration in self.trace_iterations.values():
            tmp_list = []
            for attr, _ in trace_attributes:
                if attr.startswith(PREFIX):
                    attr = attr[len(PREFIX) :]
                    value = trace_iteration.get_parameters()[attr]
                else:
                    value = getattr(trace_iteration, attr)
                if attr == "selected":
                    if value:
                        tmp_list.append("true")
                    else:
                        tmp_list.append("false")
                else:
                    tmp_list.append(value)
            data.append(tmp_list)

        arff_dict["attributes"] = trace_attributes
        arff_dict["data"] = data
        # TODO allow to pass a trace description when running a flow
        arff_dict["relation"] = "Trace"
        return arff_dict

    @classmethod
    def trace_from_arff(cls, arff_obj):
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
    def _trace_from_arff_struct(cls, attributes, content, error_message):
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
            if attribute in REQUIRED_ATTRIBUTES:
                continue
            elif attribute == "setup_string":
                continue
            elif not attribute.startswith(PREFIX):
                raise ValueError(
                    "Encountered unknown attribute %s that does not start "
                    "with prefix %s" % (attribute, PREFIX)
                )
            else:
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
                    "received: %s" % selected_value
                )

            parameters = OrderedDict(
                [(attribute, itt[attribute_idx[attribute]]) for attribute in parameter_attributes]
            )

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
    def trace_from_xml(cls, xml):
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
                    "selected field, received: %s" % selected_value
                )

            current = OpenMLTraceIteration(
                repeat, fold, iteration, setup_string, evaluation, selected,
            )
            trace[(repeat, fold, iteration)] = current

        return cls(run_id, trace)

    @classmethod
    def merge_traces(cls, traces: List["OpenMLRunTrace"]) -> "OpenMLRunTrace":

        merged_trace = (
            OrderedDict()
        )  # type: OrderedDict[Tuple[int, int, int], OpenMLTraceIteration]  # noqa E501

        previous_iteration = None
        for trace in traces:
            for iteration in trace:
                key = (iteration.repeat, iteration.fold, iteration.iteration)
                if previous_iteration is not None:
                    if list(merged_trace[previous_iteration].parameters.keys()) != list(
                        iteration.parameters.keys()
                    ):
                        raise ValueError(
                            "Cannot merge traces because the parameters are not equal: "
                            "{} vs {}".format(
                                list(merged_trace[previous_iteration].parameters.keys()),
                                list(iteration.parameters.keys()),
                            )
                        )

                if key in merged_trace:
                    raise ValueError(
                        "Cannot merge traces because key '{}' was encountered twice".format(key)
                    )

                merged_trace[key] = iteration
                previous_iteration = key

        return cls(None, merged_trace)

    def __repr__(self):
        return "[Run id: {}, {} trace iterations]".format(
            -1 if self.run_id is None else self.run_id, len(self.trace_iterations),
        )

    def __iter__(self):
        for val in self.trace_iterations.values():
            yield val


class OpenMLTraceIteration(object):
    """OpenML Trace Iteration: parsed output from Run Trace call

    Parameters
    ----------
    repeat : int
        repeat number (in case of no repeats: 0)

    fold : int
        fold number (in case of no folds: 0)

    iteration : int
        iteration number of optimization procedure

    setup_string : str
        json string representing the parameters

    evaluation : double
        The evaluation that was awarded to this trace iteration.
        Measure is defined by the task

    selected : bool
        Whether this was the best of all iterations, and hence
        selected for making predictions. Per fold/repeat there
        should be only one iteration selected

    parameters : OrderedDict
    """

    def __init__(
        self, repeat, fold, iteration, setup_string, evaluation, selected, parameters=None,
    ):

        if not isinstance(selected, bool):
            raise TypeError(type(selected))
        if setup_string and parameters:
            raise ValueError(
                "Can only be instantiated with either " "setup_string or parameters argument."
            )
        elif not setup_string and not parameters:
            raise ValueError("Either setup_string or parameters needs to be passed as " "argument.")
        if parameters is not None and not isinstance(parameters, OrderedDict):
            raise TypeError(
                "argument parameters is not an instance of OrderedDict, but %s"
                % str(type(parameters))
            )

        self.repeat = repeat
        self.fold = fold
        self.iteration = iteration
        self.setup_string = setup_string
        self.evaluation = evaluation
        self.selected = selected
        self.parameters = parameters

    def get_parameters(self):
        result = {}
        # parameters have prefix 'parameter_'

        if self.setup_string:
            for param in self.setup_string:
                key = param[len(PREFIX) :]
                value = self.setup_string[param]
                result[key] = json.loads(value)
        else:
            for param, value in self.parameters.items():
                result[param[len(PREFIX) :]] = value
        return result

    def __repr__(self):
        """
        tmp string representation, will be changed in the near future
        """
        return "[(%d,%d,%d): %f (%r)]" % (
            self.repeat,
            self.fold,
            self.iteration,
            self.evaluation,
            self.selected,
        )
