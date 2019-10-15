from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple
import webbrowser

import openml.config
from .utils import _tag_entity


class OpenMLBase(ABC):
    """ Base object for functionality that is shared across entities. """

    def __repr__(self):
        body_fields = self._get_repr_body_fields()
        return self._apply_repr_template(body_fields)

    @property
    def id(self) -> Optional[int]:
        """ The id of the entity, it is unique for its entity type. """
        from openml.datasets.dataset import OpenMLDataset
        from openml.flows.flow import OpenMLFlow
        from openml.runs.run import OpenMLRun
        from openml.study.study import BaseStudy
        from openml.tasks.task import OpenMLTask
        if isinstance(self, OpenMLDataset):
            return self.dataset_id
        if isinstance(self, OpenMLFlow):
            return self.flow_id
        if isinstance(self, OpenMLRun):
            return self.run_id
        if isinstance(self, BaseStudy):
            return self.study_id
        if isinstance(self, OpenMLTask):
            return self.task_id

    @property
    def openml_url(self) -> Optional[str]:
        """ The URL of the object on the server, if it was uploaded, else None. """
        if self.id is None:
            return None
        return self.__class__.url_for_id(self.id)

    @classmethod
    def url_for_id(cls, id_: int) -> str:
        """ Return the OpenML URL for the object of the class entity with the given id. """
        # Sample url for a flow: openml.org/f/123
        base_url = "{}".format(openml.config.server[:-len('/api/v1/xml')])
        return "{}/{}/{}".format(base_url, cls._entity_letter(), id_)

    @classmethod
    def _entity_letter(cls):
        """ Return the letter which represents the entity type in urls, e.g. 'f' for flow."""
        # We take advantage of the class naming convention (OpenMLX),
        # which holds for all entities except studies.
        from openml.study.study import BaseStudy
        from openml.tasks.task import OpenMLTask
        if issubclass(cls, BaseStudy):
            return 's'
        if issubclass(cls, OpenMLTask):
            return 't'
        return cls.__name__.lower()[len('OpenML'):][0]

    @abstractmethod
    def _get_repr_body_fields(self) -> List[Tuple[str, str]]:
        """ Collect all information to display in the __repr__ body.

        Returns
        ------
        body_fields: List[Tuple[str, str]]
            A list of (name, value) pairs to display in the body of the __repr__.
            E.g.: [('metric', 'accuracy'), ('dataset', 'iris')]
        """
        # Should be implemented in the base class.
        pass

    def _apply_repr_template(self, body_fields: List[Tuple[str, str]]) -> str:
        """ Generates the header and formats the body for string representation of the object.

         Parameters
         ----------
         body_fields: List[Tuple[str, str]]
            A list of (name, value) pairs to display in the body of the __repr__.
         """
        # We add spaces between capitals, e.g. ClassificationTask -> Classification Task
        name_with_spaces = re.sub(r"(\w)([A-Z])", r"\1 \2",
                                  self.__class__.__name__[len('OpenML'):])
        header_text = 'OpenML {}'.format(name_with_spaces)
        header = '{}\n{}\n'.format(header_text, '=' * len(header_text))

        longest_field_name_length = max(len(name) for name, value in body_fields)
        field_line_format = "{{:.<{}}}: {{}}".format(longest_field_name_length)
        body = '\n'.join(field_line_format.format(name, value) for name, value in body_fields)
        return header + body

    def open_in_browser(self):
        """ Opens the OpenML web page corresponding to this object in your default browser. """
        webbrowser.open(self.openml_url)

    def push_tag(self, tag):
        """Annotates this entity with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_entity('flow', self.id, tag)

    def remove_tag(self, tag):
        """Removes a tag from this entity on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_entity('flow', self.id, tag, untag=True)
