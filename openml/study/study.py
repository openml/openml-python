import collections
import openml
import xmltodict


class OpenMLStudy(object):

    def __init__(self, study_id, alias, main_entity_type, benchmark_suite, 
                 name, description, creation_date, creator, tags, data, tasks, 
                 flows, setups, runs):
        """
        An OpenMLStudy represents the OpenML concept of a study. It contains
        the following information: name, id, description, creation date,
        creator id and a set of tags.
    
        According to this list of tags, the study object receives a list of
        OpenML object ids (datasets, flows, tasks and setups).
    
        Can be used to obtain all relevant information from a study at once.
    
        Parameters
        ----------
        study_id : int
            the study id
        alias : str (optional)
            a string ID, unique on server (url-friendly)
        main_entity_type : str
            the entity type (e.g., task, run) that is core in this study.
            only entities of this type can be added explicitly
        benchmark_suite : int (optional)
            the benchmark suite (another study) upon which this study is ran.
            can only be active if main entity type is runs. 
        name : str
            the name of the study (meta-info)
        description : str
            brief description (meta-info)
        creation_date : str
            date of creation (meta-info)
        creator : int
            openml user id of the owner / creator
        tags : list(dict)
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
        runs : list
            a list of run ids associated with this study
        """
        self.id = study_id
        self.alias = alias
        self.main_entity_type = main_entity_type
        self.benchmark_suite = benchmark_suite
        self.name = name
        self.description = description
        self.creation_date = creation_date
        self.creator = creator
        self.tags = tags  # LEGACY. Can be removed soon
        self.data = data
        self.tasks = tasks
        self.flows = flows
        self.setups = setups
        self.runs = runs
        pass
    
    def publish(self):
        """
        Publish the study on the OpenML server.

        Returns
        -------
        study_id: int
            Id of the study uploaded to the server.
        """
        file_elements = {
            'description': self._to_xml()
        }

        return_value = openml._api_calls._perform_api_call(
            "study/",
            'post',
            file_elements=file_elements,
        )
        self.study_id = int(xmltodict.parse(return_value)['oml:study_upload']['oml:id'])
        return self.study_id
    
    def _to_xml(self):
        """Serialize object to xml for upload

        Returns
        -------
        xml_study : str
            XML description of the data.
        """
        # some can not be uploaded, e.g., id, creator, creation_date
        simple_props = ['alias', 'main_entity_type', 'name', 'description']
        # maps from attribute name (which is used as outer tag name) to immer
        # tag name (e.g., self.tasks -> <oml:tasks><oml:task_id>1987
        # </oml:task_id></oml:tasks>)
        complex_props = {
            'tasks': 'task_id',
            'runs': 'run_id',
        }

        data_container = collections.OrderedDict()
        data_dict = collections.OrderedDict([('@xmlns:oml', 'http://openml.org/openml')])
        data_container['oml:study'] = data_dict
        
        for prop_name in simple_props:
            content = getattr(self, prop_name, None)
            if content is not None:
                data_dict["oml:" + prop_name] = content
        for prop_name, inner_name in complex_props.items():
            content = getattr(self, prop_name, None)
            if content is not None:
                sub_dict = {
                    'oml:' + inner_name: content
                }
                data_dict["oml:" + prop_name] = sub_dict

        xml_string = xmltodict.unparse(
            input_dict=data_container,
            pretty=True,
        )
        # A flow may not be uploaded with the xml encoding specification:
        # <?xml version="1.0" encoding="utf-8"?>
        xml_string = xml_string.split('\n', 1)[-1]
        return xml_string
