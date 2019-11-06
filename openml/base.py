# License: BSD 3-Clause

from abc import ABC, abstractmethod
from collections import OrderedDict
import re
from typing import Optional, List, Tuple, Union, Dict
import webbrowser

import xmltodict

import openml.config
from .utils import _tag_openml_base, _get_rest_api_type_alias


class OpenMLBase(ABC):
    """ Base object for functionality that is shared across entities. """

    def __repr__(self):
        body_fields = self._get_repr_body_fields()
        return self._apply_repr_template(body_fields)

    @property
    @abstractmethod
    def id(self) -> Optional[int]:
        """ The id of the entity, it is unique for its entity type. """
        pass

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
        return "{}/{}/{}".format(openml.config.server_base_url, cls._entity_letter(), id_)

    @classmethod
    def _entity_letter(cls) -> str:
        """ Return the letter which represents the entity type in urls, e.g. 'f' for flow."""
        # We take advantage of the class naming convention (OpenMLX),
        # which holds for all entities except studies and tasks, which overwrite this method.
        return cls.__name__.lower()[len('OpenML'):][0]

    @abstractmethod
    def _get_repr_body_fields(self) -> List[Tuple[str, Union[str, int, List[str]]]]:
        """ Collect all information to display in the __repr__ body.

        Returns
        ------
        body_fields : List[Tuple[str, Union[str, int, List[str]]]]
            A list of (name, value) pairs to display in the body of the __repr__.
            E.g.: [('metric', 'accuracy'), ('dataset', 'iris')]
            If value is a List of str, then each item of the list will appear in a separate row.
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

    @abstractmethod
    def _to_dict(self) -> 'OrderedDict[str, OrderedDict]':
        """ Creates a dictionary representation of self.

        Uses OrderedDict to ensure consistent ordering when converting to xml.
        The return value (OrderedDict) will be used to create the upload xml file.
        The xml file must have the tags in exactly the order of the object's xsd.
        (see https://github.com/openml/OpenML/blob/master/openml_OS/views/pages/api_new/v1/xsd/).

        Returns
        -------
        OrderedDict
            Flow represented as OrderedDict.

        """
        # Should be implemented in the base class.
        pass

    def _to_xml(self) -> str:
        """ Generate xml representation of self for upload to server. """
        dict_representation = self._to_dict()
        xml_representation = xmltodict.unparse(dict_representation, pretty=True)

        # A task may not be uploaded with the xml encoding specification:
        # <?xml version="1.0" encoding="utf-8"?>
        encoding_specification, xml_body = xml_representation.split('\n', 1)
        return xml_body

    def _get_file_elements(self) -> Dict:
        """ Get file_elements to upload to the server, called during Publish.

        Derived child classes should overwrite this method as necessary.
        The description field will be populated automatically if not provided.
        """
        return {}

    @abstractmethod
    def _parse_publish_response(self, xml_response: Dict):
        """ Parse the id from the xml_response and assign it to self. """
        pass

    def publish(self) -> 'OpenMLBase':
        file_elements = self._get_file_elements()

        if 'description' not in file_elements:
            file_elements['description'] = self._to_xml()

        call = '{}/'.format(_get_rest_api_type_alias(self))
        response_text = openml._api_calls._perform_api_call(
            call, 'post', file_elements=file_elements
        )
        xml_response = xmltodict.parse(response_text)

        self._parse_publish_response(xml_response)
        return self

    def open_in_browser(self):
        """ Opens the OpenML web page corresponding to this object in your default browser. """
        webbrowser.open(self.openml_url)

    def push_tag(self, tag: str):
        """Annotates this entity with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_openml_base(self, tag)

    def remove_tag(self, tag: str):
        """Removes a tag from this entity on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_openml_base(self, tag, untag=True)
