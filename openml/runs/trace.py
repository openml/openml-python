import json

class OpenMLRunTrace(object):
    """OpenML Run Trace: parsed output from Run Trace call

    Parameters
    ----------
    FIXME

    """

    def __init__(self, run_id, trace_iterations):
        self.run_id = run_id
        self.trace_iterations = trace_iterations

    def __str__(self):
        return '[Run id: %d, %d trace iterations]' %(self.run_id, len(self.trace_iterations))


class OpenMLTraceIteration(object):
    """OpenML Trace Iteration: parsed output from Run Trace call

    Parameters
    ----------
    FIXME
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
        return '[(%d,%d,%d): %f (%r)]' %(self.repeat, self.fold, self.iteration,
                                          self.evaluation, self.selected)

