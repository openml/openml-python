from typing import Optional, List, Tuple

import openml.config
from .utils import _tag_entity


class OpenMLBase:
    """ Base object for functionality that is shared across entities. """
    entity_letter: str = None

    def __init__(self, entity_id: Optional[int] = None):
        self._entity_id = entity_id

    def __repr__(self):
        body_fields = self._get_repr_body_fields()
        return self._apply_repr_template(body_fields)

    @property
    def id(self) -> Optional[int]:
        """ The id of the entity, it is unique for its entity type. """
        return self._entity_id

    @property
    def openml_url(self) -> Optional[str]:
        """ The URL of the object on the server, if it was uploaded, else None. """
        if self._entity_id is None:
            return None
        return self.__class__._url_for_id(self._entity_id)

    @classmethod
    def _url_for_id(cls, id_: int) -> str:
        """ Return the OpenML URL for the object of the class entity with the given id. """
        # Sample url for a flow: openml.org/f/123
        base_url = "{}".format(openml.config.server[:-len('/api/v1/xml')])
        return "{}/{}/{}".format(base_url, cls.entity_letter, id_)

    def _get_repr_body_fields(self) -> List[Tuple[str, str]]:
        """ Collect all information to display in the __repr__ body.

        Returns
        ------
        body_fields: List[Tuple[str, str]]
            A list of (name, value) pairs to display in the body of the __repr__.
            E.g.: [('metric', 'accuracy'), ('dataset', 'iris')]
        """
        # Should be implemented in the base class.
        return []

    def _apply_repr_template(self, body_fields: List[Tuple[str, str]]) -> str:
        """ Generates the header and formats the body for string representation of the object.

         Parameters
         ----------
         body_fields: List[Tuple[str, str]]
            A list of (name, value) pairs to display in the body of the __repr__.
         """
        # Add a space in the class name, e.g. OpenMLFlow -> OpenML Flow
        entity_name = '{} {}'.format(self.__class__.__name__[:len('OpenML')],
                                     self.__class__.__name__[len('OpenML'):])
        header = '{}\n{}\n'.format(entity_name, '=' * len(entity_name))

        longest_field_name_length = max(len(name) for name, value in body_fields)
        field_line_format = "{{:.<{}}}: {{}}".format(longest_field_name_length)
        body = '\n'.join(field_line_format.format(name, value) for name, value in body_fields)
        return header + body

    def push_tag(self, tag):
        """Annotates this entity with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_entity('flow', self._entity_id, tag)

    def remove_tag(self, tag):
        """Removes a tag from this entity on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_entity('flow', self._entity_id, tag, untag=True)
