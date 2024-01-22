# License: BSD 3-Clause
from __future__ import annotations

import re
import webbrowser
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import xmltodict

import openml._api_calls
import openml.config

from .utils import _get_rest_api_type_alias, _tag_openml_base


class OpenMLBase(ABC):
    """Base object for functionality that is shared across entities."""

    def __repr__(self) -> str:
        body_fields = self._get_repr_body_fields()
        return self._apply_repr_template(body_fields)

    @property
    @abstractmethod
    def id(self) -> int | None:
        """The id of the entity, it is unique for its entity type."""

    @property
    def openml_url(self) -> str | None:
        """The URL of the object on the server, if it was uploaded, else None."""
        if self.id is None:
            return None
        return self.__class__.url_for_id(self.id)

    @classmethod
    def url_for_id(cls, id_: int) -> str:
        """Return the OpenML URL for the object of the class entity with the given id."""
        # Sample url for a flow: openml.org/f/123
        return f"{openml.config.get_server_base_url()}/{cls._entity_letter()}/{id_}"

    @classmethod
    def _entity_letter(cls) -> str:
        """Return the letter which represents the entity type in urls, e.g. 'f' for flow."""
        # We take advantage of the class naming convention (OpenMLX),
        # which holds for all entities except studies and tasks, which overwrite this method.
        return cls.__name__.lower()[len("OpenML") :][0]

    # TODO(eddiebergman): This would be much cleaner as an iterator...
    @abstractmethod
    def _get_repr_body_fields(self) -> Sequence[tuple[str, str | int | list[str] | None]]:
        """Collect all information to display in the __repr__ body.

        Returns
        -------
        body_fields : List[Tuple[str, Union[str, int, List[str]]]]
            A list of (name, value) pairs to display in the body of the __repr__.
            E.g.: [('metric', 'accuracy'), ('dataset', 'iris')]
            If value is a List of str, then each item of the list will appear in a separate row.
        """
        # Should be implemented in the base class.

    def _apply_repr_template(
        self,
        body_fields: Iterable[tuple[str, str | int | list[str] | None]],
    ) -> str:
        """Generates the header and formats the body for string representation of the object.

        Parameters
        ----------
        body_fields: List[Tuple[str, str]]
           A list of (name, value) pairs to display in the body of the __repr__.
        """
        # We add spaces between capitals, e.g. ClassificationTask -> Classification Task
        name_with_spaces = re.sub(
            r"(\w)([A-Z])",
            r"\1 \2",
            self.__class__.__name__[len("OpenML") :],
        )
        header_text = f"OpenML {name_with_spaces}"
        header = "{}\n{}\n".format(header_text, "=" * len(header_text))

        _body_fields: list[tuple[str, str | int | list[str]]] = [
            (k, "None" if v is None else v) for k, v in body_fields
        ]
        longest_field_name_length = max(len(name) for name, _ in _body_fields)
        field_line_format = f"{{:.<{longest_field_name_length}}}: {{}}"
        body = "\n".join(field_line_format.format(name, value) for name, value in _body_fields)
        return header + body

    @abstractmethod
    def _to_dict(self) -> dict[str, dict]:
        """Creates a dictionary representation of self.

        The return value will be used to create the upload xml file.
        The xml file must have the tags in exactly the order of the object's xsd.
        (see https://github.com/openml/OpenML/blob/master/openml_OS/views/pages/api_new/v1/xsd/).

        Returns
        -------
            Thing represented as dict.
        """
        # Should be implemented in the base class.

    def _to_xml(self) -> str:
        """Generate xml representation of self for upload to server."""
        dict_representation = self._to_dict()
        xml_representation = xmltodict.unparse(dict_representation, pretty=True)

        # A task may not be uploaded with the xml encoding specification:
        # <?xml version="1.0" encoding="utf-8"?>
        _encoding_specification, xml_body = xml_representation.split("\n", 1)
        return str(xml_body)

    def _get_file_elements(self) -> openml._api_calls.FILE_ELEMENTS_TYPE:
        """Get file_elements to upload to the server, called during Publish.

        Derived child classes should overwrite this method as necessary.
        The description field will be populated automatically if not provided.
        """
        return {}

    @abstractmethod
    def _parse_publish_response(self, xml_response: dict[str, str]) -> None:
        """Parse the id from the xml_response and assign it to self."""

    def publish(self) -> OpenMLBase:
        """Publish the object on the OpenML server."""
        file_elements = self._get_file_elements()

        if "description" not in file_elements:
            file_elements["description"] = self._to_xml()

        call = f"{_get_rest_api_type_alias(self)}/"
        response_text = openml._api_calls._perform_api_call(
            call,
            "post",
            file_elements=file_elements,
        )
        xml_response = xmltodict.parse(response_text)

        self._parse_publish_response(xml_response)
        return self

    def open_in_browser(self) -> None:
        """Opens the OpenML web page corresponding to this object in your default browser."""
        if self.openml_url is None:
            raise ValueError(
                "Cannot open element on OpenML.org when attribute `openml_url` is `None`",
            )

        webbrowser.open(self.openml_url)

    def push_tag(self, tag: str) -> None:
        """Annotates this entity with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_openml_base(self, tag)

    def remove_tag(self, tag: str) -> None:
        """Removes a tag from this entity on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the flow.
        """
        _tag_openml_base(self, tag, untag=True)
