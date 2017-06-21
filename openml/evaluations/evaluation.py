
class OpenMLEvaluation(object):

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

