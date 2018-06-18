
class OpenMLEvaluation(object):
    '''
    Contains all meta-information about a run / evaluation combination,
    according to the evaluation/list function

    Parameters
    ----------
    run_id : int
    
    task_id : int
    
    setup_id : int
    
    flow_id : int
    
    flow_name : str
    
    data_id : int
    
    data_name : str
        the name of the dataset
    function : str
        the evaluation function of this item (e.g., accuracy)
    upload_time : str
        the time of evaluation
    value : float
        the value of this evaluation
    array_data : str
        list of information per class (e.g., in case of precision, auroc, recall)
    '''
    def __init__(self, run_id, task_id, setup_id, flow_id, flow_name,
                 data_id, data_name, function, upload_time, value,
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
        self.array_data = array_data
