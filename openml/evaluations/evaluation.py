
class OpenMLEvaluation(object):
    """
    Contains all meta-information about a run / evaluation combination,
    according to the evaluation/list function

    Parameters
    ----------
    run_id : int
        Refers to the run.
    task_id : int
        Refers to the task.
    setup_id : int
        Refers to the setup.
    flow_id : int
        Refers to the flow.
    flow_name : str
        Name of the referred flow.
    data_id : int
        Refers to the dataset.
    data_name : str
        The name of the dataset.
    function : str
        The evaluation metric of this item (e.g., accuracy).
    upload_time : str
        The time of evaluation.
    value : float
        The value (score) of this evaluation.
    values : List[float]
        The values (scores) per repeat and fold (if requested)
    array_data : str
        list of information per class.
        (e.g., in case of precision, auroc, recall)
    """
    def __init__(self, run_id, task_id, setup_id, flow_id, flow_name,
                 data_id, data_name, function, upload_time, value, values,
                 array_data=None):
        self.run_id = run_id
        self.task_id = task_id
        self.setup_id = setup_id
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.data_id = data_id
        self.data_name = data_name
        self.function = function
        self.upload_time = upload_time
        self.value = value
        self.values = values
        self.array_data = array_data

    def __str__(self):
        object_dict = self.__dict__
        output_str = ''
        header = "OpenML Evaluation"
        header = '{}\n{}\n'.format(header, '=' * len(header))
        base_url = 'https://www.openml.org/'
        upload = '{:.<14}: {}\n'.format('Upload Date', object_dict['upload_time'])
        run = '{:.<14}: {}\n'.format('Run ID', object_dict['run_id'])
        run = run + '{:.<14}: {}\n'.format('OpenML Run URL',
                                           base_url + 'r/' + str(object_dict['run_id']))

        task = '{:.<14}: {}\n'.format('Task ID', object_dict['task_id'])
        task = task + '{:.<14}: {}\n'.format('OpenML Task URL',
                                             base_url + 't/' + str(object_dict['task_id']))

        flow = '{:.<14}: {}\n'.format('Flow ID', object_dict['flow_id'])
        flow = flow + '{:.<14}: {}\n'.format('Flow Name', object_dict['flow_name'])
        flow = flow + '{:.<14}: {}\n'.format('OpenML Flow URL',
                                             base_url + 'f/' + str(object_dict['flow_id']))

        setup = '{:.<14}: {}\n'.format('Setup ID', object_dict['setup_id'])

        data = '{:.<14}: {}\n'.format('Data ID', int(object_dict['data_id']))
        data = data + '{:.<14}: {}\n'.format('Data Name', object_dict['data_name'])
        data = data + '{:.<14}: {}\n'.format('OpenML Data URL',
                                             base_url + 'd/' + str(object_dict['data_id']))

        metric = '{:.<14}: {}\n'.format('Metric Used', object_dict['function'])
        value = '{:.<14}: {}\n'.format('Result', object_dict['value'])

        output_str = '\n' + header + upload + run + task + flow + setup + data + metric + \
            value + '\n'
        return output_str
