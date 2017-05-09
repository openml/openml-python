import json

class OpenMLRunTrace(object):
    """OpenML Run Trace: parsed output from Run Trace call

    Parameters
    ----------
    run_id : int

    trace_iterations : dict
        Mapping from key ``(repeat, fold, iteration)`` to an object of
        OpenMLTraceIteration.

    """

    def __init__(self, run_id, trace_iterations):
        self.run_id = run_id
        self.trace_iterations = trace_iterations

    def get_selected_iteration(self, fold, repeat):
        '''
        Returns the trace iteration that was marked as selected. In
        case multiple are marked as selected (should not happen) the
        first of these is returned
        '''
        for (r, f, i) in self.trace_iterations:
            if r == repeat and f == fold and self.trace_iterations[(r, f, i)].selected is True:
                return i
        raise ValueError('Could not find the selected iteration for rep/fold %d/%d' %(repeat,fold))

    def __str__(self):
        return '[Run id: %d, %d trace iterations]' %(self.run_id, len(self.trace_iterations))


class OpenMLTraceIteration(object):
    """OpenML Trace Iteration: parsed output from Run Trace call

    Parameters
    ----------
    repeat : int

    fold : int

    iteration : int

    setup_string : FIXME

    evaluation : FIXME

    selected : FIXME
    """

    def __init__(self, repeat, fold, iteration, setup_string, evaluation, selected):
        self.repeat = repeat
        self.fold = fold
        self.iteration = iteration
        self.setup_string = setup_string
        self.evaluation = evaluation
        self.selected = selected

    def get_parameters(self):
        result = {}
        # parameters have prefix 'parameter_'
        prefix = 'parameter_'

        for param in self.setup_string:
            key = param[len(prefix):]
            result[key] = json.loads(self.setup_string[param])
        return result

    def __str__(self):
        '''
        tmp string representation, will be changed in the near future 
        '''
        return '[(%d,%d,%d): %f (%r)]' %(self.repeat, self.fold, self.iteration,
                                          self.evaluation, self.selected)

