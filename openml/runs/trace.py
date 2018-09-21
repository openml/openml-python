import arff
import json
import os
import xmltodict
from collections import OrderedDict

PREFIX = 'parameter_'


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

    def get_selected_iteration(self, fold, repeat):
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
        OpenMLTraceIteration
            The trace iteration from the given fold and repeat that was
            selected as the best iteration by the search procedure
        """
        for (r, f, i) in self.trace_iterations:
            if r == repeat and f == fold and self.trace_iterations[(r, f, i)].selected is True:
                return i
        raise ValueError('Could not find the selected iteration for rep/fold %d/%d' % (repeat, fold))

    @classmethod
    def generate(cls, attributes, content):
        """Generates an OpenMLRunTrace.

        Generates the trace object from the attributes and content extracted
        while running the underlying flow.

        Returns
        -------
        OpenMLRunTrace
        """

        if content is None:
            if attributes is None:
                return None
            else:
                raise ValueError('Trace content not available.')
        if len(content) == 0:
            raise ValueError('Trace content is empty.')
        if attributes is not None:
            if len(attributes) != len(content[0]):
                raise ValueError(
                    'Trace_attributes and trace_content not compatible:'
                    ' %s vs %s' % (attributes, content[0])
                )
        else:
            raise ValueError('No trace attributes available')

        trace = OrderedDict()
        attribute_idx = {att[0]: idx for idx, att in enumerate(attributes)}

        required_attributes = ['repeat', 'fold', 'iteration', 'evaluation',
                               'selected']
        for required_attribute in required_attributes:
            if required_attribute not in attribute_idx:
                raise ValueError(
                    'arff misses required attribute: %s' % required_attribute)
        parameter_attributes = []
        for attribute in attribute_idx:
            if (
                attribute not in required_attributes
                and attribute != 'setup_string'
            ):
                parameter_attributes.append(attribute)

        for itt in content:
            repeat = int(itt[attribute_idx['repeat']])
            fold = int(itt[attribute_idx['fold']])
            iteration = int(itt[attribute_idx['iteration']])
            evaluation = float(itt[attribute_idx['evaluation']])
            selected_value = itt[attribute_idx['selected']]
            if 'setup_string' in attribute_idx:
                raise ValueError(
                    'setup_string not allowed when constructing a trace object'
                    ' run results.'
                )
            setup_string = None
            if selected_value == 'true':
                selected = True
            elif selected_value == 'false':
                selected = False
            else:
                raise ValueError(
                    'expected {"true", "false"} value for selected field, '
                    'received: %s' % selected_value
                )
            parameters = OrderedDict()
            for attribute in parameter_attributes:
                parameters[attribute] = itt[attribute_idx[attribute]]

            current = OpenMLTraceIteration(
                repeat,
                fold,
                iteration,
                setup_string,
                evaluation,
                selected,
                paramaters=parameters
            )
            trace[(repeat, fold, iteration)] = current

        return cls(None, trace)

    @classmethod
    def from_filesystem(cls, file_path):
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
            raise ValueError('Trace file doesn\'t exist')

        with open(file_path, 'r') as fp:
            trace_arff = arff.load(fp)

        for trace_idx in range(len(trace_arff['data'])):
            # iterate over first three entrees of a trace row
            # (fold, repeat, trace_iteration) these should be int
            for line_idx in range(3):
                trace_arff['data'][trace_idx][line_idx] = int(
                    trace_arff['data'][trace_idx][line_idx]
                )

        return cls.trace_from_arff(trace_arff)

    def to_filesystem(self, file_path):
        """Serialize the trace object to the filesystem.

        Serialize the trace object as an arff.

        Parameters
        ----------
        file_path: str
            File path where the trace arff will be stored.
        """

        trace_arff = arff.dumps(self.trace_to_arff())
        with open(os.path.join(file_path, 'trace.arff'), 'w') as f:
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
            ('repeat', 'NUMERIC'),
            ('fold', 'NUMERIC'),
            ('iteration', 'NUMERIC'),
            ('evaluation', 'NUMERIC'),
            ('selected', ['true', 'false']),
        ]
        for parameter in next(
            iter(self.trace_iterations.values())
        ).get_parameters():
            trace_attributes.append(('parameter_' + parameter, 'STRING'))

        arff_dict = OrderedDict()
        data = []
        for trace_iteration in self.trace_iterations.values():
            tit_list = []
            for attr, _ in trace_attributes:
                if attr.startswith('parameter_'):
                    attr = attr[len('parameter_'):]
                    value = trace_iteration.get_parameters()[attr]
                else:
                    value = getattr(trace_iteration, attr)
                if attr == 'selected':
                    if value:
                        tit_list.append('true')
                    else:
                        tit_list.append('false')
                else:
                    tit_list.append(value)
            data.append(tit_list)

        arff_dict['attributes'] = trace_attributes
        arff_dict['data'] = data
        arff_dict['relation'] = "Trace"
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
        trace = OrderedDict()
        # flag if setup string is in attributes
        attribute_idx = {
            att[0]: idx for idx, att in enumerate(arff_obj['attributes'])
        }
        required_attributes = [
            'repeat',
            'fold',
            'iteration',
            'evaluation',
            'selected',
        ]
        for required_attribute in required_attributes:
            if required_attribute not in attribute_idx:
                raise ValueError(
                    'arff misses required attribute: %s' % required_attribute
                )
        if 'setup_string' in attribute_idx:
            raise ValueError(
                'setup_string not supported for arff serialization'
            )

        for itt in arff_obj['data']:
            repeat = int(itt[attribute_idx['repeat']])
            fold = int(itt[attribute_idx['fold']])
            iteration = int(itt[attribute_idx['iteration']])
            evaluation = float(itt[attribute_idx['evaluation']])
            selected_value = itt[attribute_idx['selected']]
            if selected_value == 'true':
                selected = True
            elif selected_value == 'false':
                selected = False
            else:
                raise ValueError(
                    'expected {"true", "false"} value for selected field, '
                    'received: %s' % selected_value
                )

            parameters = OrderedDict()
            for attribute_name, idx in attribute_idx.items():
                if attribute_name in required_attributes:
                    continue
                if attribute_name.startswith(PREFIX):
                    parameters[attribute_name] = itt[idx]


            current = OpenMLTraceIteration(
                repeat=repeat,
                fold=fold,
                iteration=iteration,
                setup_string=None,
                evaluation=evaluation,
                selected=selected,
                paramaters=parameters,
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
            Object containing run id and a dict containing the trace iterations.
        """
        result_dict = xmltodict.parse(
            xml, force_list=('oml:trace_iteration',)
        )['oml:trace']

        run_id = result_dict['oml:run_id']
        trace = OrderedDict()

        if 'oml:trace_iteration' not in result_dict:
            raise ValueError('Run does not contain valid trace. ')
        if not isinstance(result_dict['oml:trace_iteration'], list):
            raise TypeError(type(result_dict['oml:trace_iteration']))

        for itt in result_dict['oml:trace_iteration']:
            repeat = int(itt['oml:repeat'])
            fold = int(itt['oml:fold'])
            iteration = int(itt['oml:iteration'])
            setup_string = json.loads(itt['oml:setup_string'])
            evaluation = float(itt['oml:evaluation'])
            selected_value = itt['oml:selected']
            if selected_value == 'true':
                selected = True
            elif selected_value == 'false':
                selected = False
            else:
                raise ValueError(
                    'expected {"true", "false"} value for '
                    'selected field, received: %s' % selected_value
                )

            current = OpenMLTraceIteration(
                repeat,
                fold,
                iteration,
                setup_string,
                evaluation,
                selected,
            )
            trace[(repeat, fold, iteration)] = current

        return cls(run_id, trace)

    def __str__(self):
        return '[Run id: %d, %d trace iterations]' % (
            self.run_id,
            len(self.trace_iterations),
        )


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
        self,
        repeat,
        fold,
        iteration,
        setup_string,
        evaluation,
        selected,
        paramaters=None,
    ):

        if not isinstance(selected, bool):
            raise TypeError(type(selected))
        if setup_string and paramaters:
            raise ValueError(
                'Can only be instantiated with either '
                'setup_string or parameters argument.'
            )
        elif not setup_string and not paramaters:
            raise ValueError(
                'Either setup_string or parameters needs to be paased as '
                'argument.'
            )
        if paramaters is not None and not isinstance(paramaters, OrderedDict):
            raise TypeError(
                'argument parameters is not an instance of OrderedDict, but %s'
                % str(type(paramaters))
            )

        self.repeat = repeat
        self.fold = fold
        self.iteration = iteration
        self.setup_string = setup_string
        self.evaluation = evaluation
        self.selected = selected
        self.parameters = paramaters

    def get_parameters(self):
        if self.parameters:
            return self.parameters

        result = {}
        # parameters have prefix 'parameter_'

        prefix = 'parameter_'
        for param in self.setup_string:
            key = param[len(prefix):]
            value = self.setup_string[param]
            result[key] = json.loads(value)
        return result

    def __str__(self):
        """
        tmp string representation, will be changed in the near future 
        """
        return '[(%d,%d,%d): %f (%r)]' % (
            self.repeat,
            self.fold,
            self.iteration,
            self.evaluation,
            self.selected,
        )
