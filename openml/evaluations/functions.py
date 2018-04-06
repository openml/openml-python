import xmltodict

from openml.exceptions import OpenMLServerNoResult
import openml.utils
import openml._api_calls
from ..evaluations import OpenMLEvaluation


def list_evaluations(function, offset=None, size=None, id=None, task=None,
                     setup=None, flow=None, uploader=None, tag=None):
    """
    List all run-evaluation pairs matching all of the given filters.
    (Supports large amount of results)

    Parameters
    ----------
    function : str
        the evaluation function. e.g., predictive_accuracy
    offset : int, optional
        the number of runs to skip, starting from the first
    size : int, optional
        the maximum number of runs to show

    id : list, optional

    task : list, optional

    setup: list, optional

    flow : list, optional

    uploader : list, optional

    tag : str, optional

    Returns
    -------
    dict
    """

    return openml.utils.list_all(_list_evaluations, function, offset=offset, size=size,
                                 id=id, task=task, setup=setup, flow=flow, uploader=uploader, tag=tag)


def _list_evaluations(function, id=None, task=None,
                      setup=None, flow=None, uploader=None, **kwargs):
    """
    Perform API call ``/evaluation/function{function}/{filters}``

    Parameters
    ----------
    The arguments that are lists are separated from the single value
    ones which are put into the kwargs.

    function : str
        the evaluation function. e.g., predictive_accuracy

    id : list, optional

    task : list, optional

    setup: list, optional

    flow : list, optional

    uploader : list, optional

    kwargs: dict, optional
        Legal filter operators: tag, limit, offset.

    Returns
    -------
    dict
    """

    api_call = "evaluation/list/function/%s" % function
    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)
    if id is not None:
        api_call += "/run/%s" % ','.join([str(int(i)) for i in id])
    if task is not None:
        api_call += "/task/%s" % ','.join([str(int(i)) for i in task])
    if setup is not None:
        api_call += "/setup/%s" % ','.join([str(int(i)) for i in setup])
    if flow is not None:
        api_call += "/flow/%s" % ','.join([str(int(i)) for i in flow])
    if uploader is not None:
        api_call += "/uploader/%s" % ','.join([str(int(i)) for i in uploader])

    return __list_evaluations(api_call)


def __list_evaluations(api_call):
    """Helper function to parse API calls which are lists of runs"""
    xml_string = openml._api_calls._perform_api_call(api_call)
    evals_dict = xmltodict.parse(xml_string, force_list=('oml:evaluation',))
    # Minimalistic check if the XML is useful
    if 'oml:evaluations' not in evals_dict:
        raise ValueError('Error in return XML, does not contain "oml:evaluations": %s'
                         % str(evals_dict))

    assert type(evals_dict['oml:evaluations']['oml:evaluation']) == list, \
        type(evals_dict['oml:evaluations'])

    evals = dict()
    for eval_ in evals_dict['oml:evaluations']['oml:evaluation']:
        run_id = int(eval_['oml:run_id'])
        array_data = None
        if 'oml:array_data' in eval_:
            eval_['oml:array_data']

        evaluation = OpenMLEvaluation(int(eval_['oml:run_id']), int(eval_['oml:task_id']),
                                      int(eval_['oml:setup_id']), int(eval_['oml:flow_id']),
                                      eval_['oml:flow_name'], eval_['oml:data_id'],
                                      eval_['oml:data_name'], eval_['oml:function'],
                                      eval_['oml:upload_time'], float(eval_['oml:value']),
                                      array_data)
        evals[run_id] = evaluation
    return evals
