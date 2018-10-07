import openml
from openml.flows import OpenMLFlow
from openml.setups import OpenMLParameter


def openml_param_name_to_sklearn(openml_parameter, flow):
    """
    Converts the name of an OpenMLParameter into the sklean name, given a flow.
    Note that the same parameter might have a different name in various flows
    (e.g., the parameter `min_num_splits` will be called `min_num_splits` in
    a `DecisionTreeClassifier`, but `base_estimator__min_num_splits`) when the
    `DecisionTreeClassifier` is wrapped in `AdaboostClassifier`.

    Parameters
    ----------
    openml_parameter: OpenMLParameter
        The parameter under consideration

    flow: OpenMLFlow
        The flow that provides context.
    """
    if not isinstance(openml_parameter, OpenMLParameter):
        raise ValueError('openml_parameter should be an instance of '
                         'OpenMLParameter')
    if not isinstance(flow, OpenMLFlow):
        raise ValueError('flow should be an instance of OpenMLFlow')

    flow_structure = openml.flows.flow_structure(flow, 'name')
    complete = flow_structure[openml_parameter.flow_name] + \
               [openml_parameter.parameter_name]
    return '__'.join(complete)
