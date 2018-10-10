from openml.flows import OpenMLFlow
from openml.setups import OpenMLParameter


def openml_param_name_to_sklearn(openml_parameter, flow):
    """
    Converts the name of an OpenMLParameter into the sklean name, given a flow.

    Parameters
    ----------
    openml_parameter: OpenMLParameter
        The parameter under consideration

    flow: OpenMLFlow
        The flow that provides context.

    Returns
    -------
    sklearn_parameter_name: str
        The name the parameter will have once used in scikit-learn
    """
    if not isinstance(openml_parameter, OpenMLParameter):
        raise ValueError('openml_parameter should be an instance of '
                         'OpenMLParameter')
    if not isinstance(flow, OpenMLFlow):
        raise ValueError('flow should be an instance of OpenMLFlow')

    flow_structure = flow.get_structure('name')
    if openml_parameter.flow_name not in flow_structure:
        raise ValueError('Obtained OpenMLParameter and OpenMLFlow do not '
                         'correspond. ')

    return '__'.join(flow_structure[openml_parameter.flow_name] +
                     [openml_parameter.parameter_name])
