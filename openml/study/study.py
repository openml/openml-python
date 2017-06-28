
class OpenMLStudy(object):
    '''
    An OpenMLStudy represents the OpenML concept of a study. It contains
    the following information: name, id, description, creation date,
    creator id and a set of tags.

    According to this list of tags, the study object receives a list of
    OpenML object ids (datasets, flows, tasks and setups).

    Can be used to obtain all relevant information from a study at once.

    Parameters
       ----------
        id : int
            the study id
        name : str
            the name of the study (meta-info)
        description : str
            brief description (meta-info)
        creation_date : str
            date of creation (meta-info)
        creator : int
            openml user id of the owner / creator
        tag : list(dict)
            The list of tags shows which tags are associated with the study.
            Each tag is a dict of (tag) name, window_start and write_access.
        data : list
            a list of data ids associated with this study
        tasks : list
            a list of task ids associated with this study
        flows : list
            a list of flow ids associated with this study
        setups : list
            a list of setup ids associated with this study
    '''

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

