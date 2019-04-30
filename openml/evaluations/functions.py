import json
import xmltodict
import pandas as pd
from typing import Union, List, Optional, Dict

import openml.utils
import openml._api_calls
from ..evaluations import OpenMLEvaluation


def list_evaluations(
    function: str,
    offset: Optional[int] = None,
    size: Optional[int] = None,
    id: Optional[List] = None,
    task: Optional[List] = None,
    setup: Optional[List] = None,
    flow: Optional[List] = None,
    uploader: Optional[List] = None,
    tag: Optional[str] = None,
    per_fold: Optional[bool] = None,
    output_format: str = 'object'
) -> Union[Dict, pd.DataFrame]:
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

    per_fold : bool, optional

    output_format: str, optional (default='object')
        The parameter decides the format of the output.
        - If 'object' the output is a dict of OpenMLEvaluation objects
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict or dataframe
    """
    if output_format not in ['dataframe', 'dict', 'object']:
        raise ValueError("Invalid output format selected. "
                         "Only 'object', 'dataframe', or 'dict' applicable.")

    per_fold_str = None
    if per_fold is not None:
        per_fold_str = str(per_fold).lower()

    return openml.utils._list_all(output_format=output_format,
                                  listing_call=_list_evaluations,
                                  function=function,
                                  offset=offset,
                                  size=size,
                                  id=id,
                                  task=task,
                                  setup=setup,
                                  flow=flow,
                                  uploader=uploader,
                                  tag=tag,
                                  per_fold=per_fold_str)


def _list_evaluations(
    function: str,
    id: Optional[List] = None,
    task: Optional[List] = None,
    setup: Optional[List] = None,
    flow: Optional[List] = None,
    uploader: Optional[List] = None,
    output_format: str = 'object',
    **kwargs
) -> Union[Dict, pd.DataFrame]:
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

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict of objects, or dataframe
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

    return __list_evaluations(api_call, output_format=output_format)


def __list_evaluations(api_call, output_format='object'):
    """Helper function to parse API calls which are lists of runs"""
    xml_string = openml._api_calls._perform_api_call(api_call, 'get')
    evals_dict = xmltodict.parse(xml_string, force_list=('oml:evaluation',))
    # Minimalistic check if the XML is useful
    if 'oml:evaluations' not in evals_dict:
        raise ValueError('Error in return XML, does not contain '
                         '"oml:evaluations": %s' % str(evals_dict))

    assert type(evals_dict['oml:evaluations']['oml:evaluation']) == list, \
        type(evals_dict['oml:evaluations'])

    evals = dict()
    for eval_ in evals_dict['oml:evaluations']['oml:evaluation']:
        run_id = int(eval_['oml:run_id'])
        value = None
        values = None
        array_data = None
        if 'oml:value' in eval_:
            value = float(eval_['oml:value'])
        if 'oml:values' in eval_:
            values = json.loads(eval_['oml:values'])
        if 'oml:array_data' in eval_:
            array_data = eval_['oml:array_data']

        if output_format == 'object':
            evals[run_id] = OpenMLEvaluation(int(eval_['oml:run_id']),
                                             int(eval_['oml:task_id']),
                                             int(eval_['oml:setup_id']),
                                             int(eval_['oml:flow_id']),
                                             eval_['oml:flow_name'],
                                             eval_['oml:data_id'],
                                             eval_['oml:data_name'],
                                             eval_['oml:function'],
                                             eval_['oml:upload_time'],
                                             value, values, array_data)
        else:
            # for output_format in ['dict', 'dataframe']
            evals[run_id] = {'run_id': int(eval_['oml:run_id']),
                             'task_id': int(eval_['oml:task_id']),
                             'setup_id': int(eval_['oml:setup_id']),
                             'flow_id': int(eval_['oml:flow_id']),
                             'flow_name': eval_['oml:flow_name'],
                             'data_id': eval_['oml:data_id'],
                             'data_name': eval_['oml:data_name'],
                             'function': eval_['oml:function'],
                             'upload_time': eval_['oml:upload_time'],
                             'value': value,
                             'values': values,
                             'array_data': array_data}

    if output_format == 'dataframe':
        evals = pd.DataFrame.from_dict(evals, orient='index')

    return evals
