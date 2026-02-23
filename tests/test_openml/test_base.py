# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.base module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import xmltodict

from openml.base import OpenMLBase


# Concrete implementation for testing the abstract base class
class OpenMLTest(OpenMLBase):
    """Concrete implementation of OpenMLBase for testing."""

    def __init__(self, entity_id: int | None = None):
        self._id = entity_id
        self.name = "TestEntity"
        self.description = "Test Description"
        self.some_value = 42

    @property
    def id(self) -> int | None:
        return self._id

    def _get_repr_body_fields(self):
        return [
            ("Name", self.name),
            ("Description", self.description),
            ("Some Value", self.some_value),
            ("None Field", None),
        ]

    def _to_dict(self) -> dict[str, dict]:
        return {
            "oml:test_entity": {
                "@xmlns:oml": "http://openml.org/openml",
                "oml:name": self.name,
                "oml:description": self.description,
                "oml:value": self.some_value,
            }
        }

    def _parse_publish_response(self, xml_response: dict[str, str]) -> None:
        self._id = int(xml_response["oml:upload_test"]["oml:id"])


class TestOpenMLBase:
    """Test suite for OpenMLBase class."""

    @pytest.fixture
    def concrete_instance(self):
        """Create a concrete instance for testing."""
        return OpenMLTest(entity_id=123)

    @pytest.fixture
    def concrete_instance_no_id(self):
        """Create a concrete instance without ID."""
        return OpenMLTest(entity_id=None)

    def test_repr(self, concrete_instance):
        """Test __repr__ method produces correct output."""
        repr_str = repr(concrete_instance)
        
        assert "OpenML Test" in repr_str
        assert "Name" in repr_str
        assert "TestEntity" in repr_str
        assert "Description" in repr_str
        assert "Test Description" in repr_str
        assert "Some Value" in repr_str
        assert "42" in repr_str
        assert "None Field" in repr_str
        assert "None" in repr_str

    def test_repr_with_list_value(self):
        """Test __repr__ with list values in body fields."""
        class ListFieldEntity(OpenMLTest):
            def _get_repr_body_fields(self):
                return [
                    ("Tags", ["tag1", "tag2", "tag3"]),
                    ("Name", "TestList"),
                ]
        
        entity = ListFieldEntity()
        repr_str = repr(entity)
        
        assert "Tags" in repr_str
        assert "['tag1', 'tag2', 'tag3']" in repr_str

    def test_id_property(self, concrete_instance):
        """Test id property returns correct value."""
        assert concrete_instance.id == 123

    def test_id_property_none(self, concrete_instance_no_id):
        """Test id property when None."""
        assert concrete_instance_no_id.id is None

    def test_openml_url_with_id(self, concrete_instance):
        """Test openml_url property when id is set."""
        with patch("openml.config.get_server_base_url") as mock_base_url:
            mock_base_url.return_value = "https://test.openml.org/api/v1/xml"
            url = concrete_instance.openml_url
            
            assert url == "https://test.openml.org/api/v1/xml/t/123"

    def test_openml_url_without_id(self, concrete_instance_no_id):
        """Test openml_url property when id is None."""
        assert concrete_instance_no_id.openml_url is None

    def test_url_for_id_class_method(self):
        """Test url_for_id class method."""
        with patch("openml.config.get_server_base_url") as mock_base_url:
            mock_base_url.return_value = "https://openml.org"
            url = OpenMLTest.url_for_id(456)
            
            assert url == "https://openml.org/t/456"

    def test_entity_letter(self):
        """Test _entity_letter method returns correct letter."""
        # For OpenMLTest, it should extract 't' from the name
        letter = OpenMLTest._entity_letter()
        assert letter == "t"

    def test_entity_letter_different_class(self):
        """Test _entity_letter with different class names."""
        class OpenMLTask(OpenMLBase):
            @property
            def id(self):
                return 1
            def _get_repr_body_fields(self):
                return []
            def _to_dict(self):
                return {}
            def _parse_publish_response(self, xml_response):
                pass
        
        assert OpenMLTask._entity_letter() == "t"

    def test_apply_repr_template(self, concrete_instance):
        """Test _apply_repr_template method."""
        body_fields = [
            ("Field1", "Value1"),
            ("LongerFieldName", "Value2"),
            ("Field3", 123),
        ]
        
        result = concrete_instance._apply_repr_template(body_fields)
        
        assert "OpenML Test" in result
        assert "Field1" in result
        assert "Value1" in result
        assert "LongerFieldName" in result
        assert "Value2" in result
        assert "123" in result
        # Check for dots alignment
        assert "..." in result

    def test_apply_repr_template_with_none_values(self, concrete_instance):
        """Test _apply_repr_template converts None to string."""
        body_fields = [("Field", None)]
        result = concrete_instance._apply_repr_template(body_fields)
        
        assert "None" in result

    def test_to_xml(self, concrete_instance):
        """Test _to_xml method generates valid XML."""
        xml_output = concrete_instance._to_xml()
        
        # Should not contain the XML declaration
        assert "<?xml" not in xml_output
        # Should contain the entity data
        assert "test_entity" in xml_output
        assert "TestEntity" in xml_output
        assert "Test Description" in xml_output

    def test_to_xml_valid_format(self, concrete_instance):
        """Test _to_xml produces parseable XML."""
        xml_output = concrete_instance._to_xml()
        
        # Should be parseable by xmltodict
        parsed = xmltodict.parse(xml_output)
        assert "oml:test_entity" in parsed

    def test_get_file_elements_default(self, concrete_instance):
        """Test _get_file_elements returns empty dict by default."""
        file_elements = concrete_instance._get_file_elements()
        
        assert file_elements == {}
        assert isinstance(file_elements, dict)

    def test_publish_success(self, concrete_instance_no_id):
        """Test publish method successfully publishes entity."""
        mock_response = """
        <oml:upload_test>
            <oml:id>999</oml:id>
        </oml:upload_test>
        """
        
        with patch("openml._api_calls._perform_api_call") as mock_api_call, \
             patch("xmltodict.parse") as mock_parse, \
             patch("openml.base._get_rest_api_type_alias") as mock_alias:
            mock_api_call.return_value = mock_response
            mock_parse.return_value = {"oml:upload_test": {"oml:id": "999"}}
            mock_alias.return_value = "test"
            
            result = concrete_instance_no_id.publish()
            
            # Check that ID was set
            assert result.id == 999
            assert result == concrete_instance_no_id
            # Check API was called correctly
            mock_api_call.assert_called_once()

    def test_publish_with_custom_file_elements(self):
        """Test publish with custom file elements."""
        class CustomFileEntity(OpenMLTest):
            def _get_file_elements(self):
                return {"custom_file": "file_content"}
        
        entity = CustomFileEntity()
        mock_response = """
        <oml:upload_test>
            <oml:id>888</oml:id>
        </oml:upload_test>
        """
        
        with patch("openml._api_calls._perform_api_call") as mock_api_call, \
             patch("xmltodict.parse") as mock_parse, \
             patch("openml.base._get_rest_api_type_alias") as mock_alias:
            mock_api_call.return_value = mock_response
            mock_parse.return_value = {"oml:upload_test": {"oml:id": "888"}}
            mock_alias.return_value = "test"
            
            entity.publish()
            
            # Check that file_elements includes custom file and description
            call_args = mock_api_call.call_args
            file_elements = call_args[1]["file_elements"]
            assert "custom_file" in file_elements
            assert "description" in file_elements

    def test_open_in_browser_with_id(self, concrete_instance):
        """Test open_in_browser opens URL when ID is set."""
        with patch("webbrowser.open") as mock_browser, \
             patch("openml.config.get_server_base_url") as mock_base_url:
            mock_base_url.return_value = "https://openml.org"
            
            concrete_instance.open_in_browser()
            
            mock_browser.assert_called_once_with("https://openml.org/t/123")

    def test_open_in_browser_without_id(self, concrete_instance_no_id):
        """Test open_in_browser raises error when ID is None."""
        with pytest.raises(ValueError, match="Cannot open element on OpenML.org"):
            concrete_instance_no_id.open_in_browser()

    def test_push_tag(self, concrete_instance):
        """Test push_tag method calls tag utility function."""
        with patch("openml.base._tag_openml_base") as mock_tag:
            concrete_instance.push_tag("test_tag")
            
            mock_tag.assert_called_once_with(concrete_instance, "test_tag")

    def test_remove_tag(self, concrete_instance):
        """Test remove_tag method calls tag utility function with untag=True."""
        with patch("openml.base._tag_openml_base") as mock_tag:
            concrete_instance.remove_tag("test_tag")
            
            mock_tag.assert_called_once_with(concrete_instance, "test_tag", untag=True)


class TestOpenMLBaseEdgeCases:
    """Test edge cases and error conditions."""

    def test_repr_empty_body_fields(self):
        """Test __repr__ with empty body fields."""
        class EmptyEntity(OpenMLTest):
            def _get_repr_body_fields(self):
                return []
        
        entity = EmptyEntity()
        # Should raise error because max() on empty sequence
        with pytest.raises(ValueError):
            repr(entity)

    def test_repr_single_field(self):
        """Test __repr__ with single field."""
        class SingleFieldEntity(OpenMLTest):
            def _get_repr_body_fields(self):
                return [("OnlyField", "OnlyValue")]
        
        entity = SingleFieldEntity()
        repr_str = repr(entity)
        
        assert "OnlyField" in repr_str
        assert "OnlyValue" in repr_str

    def test_to_dict_complex_structure(self):
        """Test _to_dict with nested structure."""
        class ComplexEntity(OpenMLTest):
            def _to_dict(self):
                return {
                    "oml:complex": {
                        "@xmlns:oml": "http://openml.org/openml",
                        "oml:nested": {
                            "oml:value": 123,
                            "oml:name": "nested_name",
                        },
                        "oml:list": ["item1", "item2"],
                    }
                }
        
        entity = ComplexEntity()
        result = entity._to_dict()
        
        assert "oml:complex" in result
        assert "oml:nested" in result["oml:complex"]
        assert result["oml:complex"]["oml:nested"]["oml:value"] == 123

    def test_xml_special_characters(self):
        """Test XML generation with special characters."""
        class SpecialCharEntity(OpenMLTest):
            def _to_dict(self):
                return {
                    "oml:entity": {
                        "@xmlns:oml": "http://openml.org/openml",
                        "oml:description": "Test <>&\"' special chars",
                    }
                }
        
        entity = SpecialCharEntity()
        xml_str = entity._to_xml()
        
        # xmltodict should escape special characters
        assert "Test" in xml_str

    def test_publish_api_call_parameters(self):
        """Test that publish passes correct parameters to API call."""
        entity = OpenMLTest()
        mock_response = """
        <oml:upload_test>
            <oml:id>777</oml:id>
        </oml:upload_test>
        """
        
        with patch("openml._api_calls._perform_api_call") as mock_api_call, \
             patch("openml.base._get_rest_api_type_alias") as mock_alias, \
             patch("xmltodict.parse") as mock_parse:
            mock_api_call.return_value = mock_response
            mock_alias.return_value = "test_entity"
            mock_parse.return_value = {"oml:upload_test": {"oml:id": "777"}}
            
            entity.publish()
            
            # Check call arguments
            call_args = mock_api_call.call_args
            assert call_args[0][0] == "test_entity/"
            assert call_args[0][1] == "post"
            assert "file_elements" in call_args[1]

    def test_parse_publish_response_with_different_structure(self):
        """Test _parse_publish_response with different XML structure."""
        class DifferentResponseEntity(OpenMLTest):
            def _parse_publish_response(self, xml_response):
                self._id = int(xml_response["oml:custom_upload"]["oml:entity_id"])
        
        entity = DifferentResponseEntity()
        xml_response = {
            "oml:custom_upload": {
                "oml:entity_id": "555"
            }
        }
        
        entity._parse_publish_response(xml_response)
        assert entity.id == 555
