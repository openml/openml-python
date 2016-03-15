class OpenMLRun(object):
    def __init__(self, run_id, uploader, uploader_name, task_id, task_type,
                 task_evaluation_measure, flow_id, flow_name,  setup_id,
                 setup_string, parameter_settings, dataset_id,
                 predictions_url, evaluations):
        self.run_id = run_id
        self.uploader = uploader
        self.uploader_name = uploader_name
        self.task_id = task_id
        self.task_type = task_type
        self.task_evaluation_measure = task_evaluation_measure
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.setup_id = setup_id
        self.setup_string = setup_string
<<<<<<< HEAD
        self.tags = tags
        self.datasets = datasets
        self.files = files
        self.evaluations = evaluations
=======
        self.parameter_settings = parameter_settings
        self.dataset_id = dataset_id
        self.predictions_url = predictions_url
        self.evaluations = evaluations
>>>>>>> ADD download run functionality
