
class OpenMLStudy(object):

    def __init__(self, id, name, description, creation_date, creator,
                 tag, data, tasks, flows, setups):
        self.id = id
        self.name = name
        self.description = description
        self.creation_date = creation_date
        self.creator = creator
        self.tag = tag
        self.data = data
        self.tasks = tasks
        self.flows = flows
        self.setups = setups
        pass

