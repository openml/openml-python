class OpenMLRun(object):
    def __init__(self, run_id, uploader, task_id, flow_id, setup_string,
                 setup_id, tags, datasets, files, evaluations):
        self.run_id = run_id
        self.uploader = uploader
        self.task_id = task_id
        self.flow_id = flow_id
        self.setup_id = setup_id
        self.setup_string = setup_string
        self.tags = tags
        self.datasets = datasets
        self.files = files
        self.evaluations = evaluations
