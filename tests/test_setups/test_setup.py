# License: BSD 3-Clause
from __future__ import annotations

import random

import pytest

import openml
from openml.base import OpenMLBase
from openml.setups.setup import OpenMLSetup
from openml.testing import TestBase
from openml.utils import _tag_entity


class TestOpenMLSetup(TestBase):
    """Tests for OpenMLSetup inheriting OpenMLBase and tagging support."""

    def test_setup_is_openml_base(self):
        """OpenMLSetup should be a subclass of OpenMLBase."""
        setup = OpenMLSetup(setup_id=1, flow_id=100, parameters=None)
        assert isinstance(setup, OpenMLBase)

    def test_setup_id_property(self):
        """The id property should return setup_id."""
        setup = OpenMLSetup(setup_id=42, flow_id=100, parameters=None)
        assert setup.id == 42
        assert setup.id == setup.setup_id

    def test_setup_repr(self):
        """The repr should use OpenMLBase format and contain expected fields."""
        setup = OpenMLSetup(setup_id=1, flow_id=100, parameters=None)
        repr_str = repr(setup)
        assert "OpenML Setup" in repr_str
        assert "Setup ID" in repr_str
        assert "Flow ID" in repr_str

    def test_setup_repr_with_parameters(self):
        """The repr should show parameter count when parameters are present."""
        # Create a minimal mock parameter-like dict
        setup = OpenMLSetup(setup_id=1, flow_id=100, parameters={1: "a", 2: "b"})
        repr_str = repr(setup)
        assert "# of Parameters" in repr_str

    def test_setup_publish_raises(self):
        """Calling publish() on a setup should raise NotImplementedError."""
        setup = OpenMLSetup(setup_id=1, flow_id=100, parameters=None)
        with pytest.raises(NotImplementedError, match="Setups cannot be published"):
            setup.publish()

    def test_setup_parse_publish_response_raises(self):
        """Calling _parse_publish_response should raise NotImplementedError."""
        setup = OpenMLSetup(setup_id=1, flow_id=100, parameters=None)
        with pytest.raises(NotImplementedError, match="Setups cannot be published"):
            setup._parse_publish_response({})

    def test_setup_openml_url(self):
        """The openml_url property should return a valid URL."""
        setup = OpenMLSetup(setup_id=1, flow_id=100, parameters=None)
        url = setup.openml_url
        assert url is not None
        assert "/s/1" in url

    def test_setup_validation(self):
        """Existing validation in __post_init__ should still work."""
        with pytest.raises(ValueError, match="setup id should be int"):
            OpenMLSetup(setup_id="not_an_int", flow_id=100, parameters=None)

        with pytest.raises(ValueError, match="flow id should be int"):
            OpenMLSetup(setup_id=1, flow_id="not_an_int", parameters=None)

        with pytest.raises(ValueError, match="parameters should be dict"):
            OpenMLSetup(setup_id=1, flow_id=100, parameters="not_a_dict")

    @pytest.mark.test_server()
    def test_tag_untag_setup_via_entity(self):
        """Test tagging and untagging a setup via _tag_entity."""
        # Setup ID 1 should exist on the test server
        tag = "test_setup_tag_%d" % random.randint(1, 1_000_000)
        all_tags = _tag_entity("setup", 1, tag)
        assert tag in all_tags
        all_tags = _tag_entity("setup", 1, tag, untag=True)
        assert tag not in all_tags

    @pytest.mark.test_server()
    def test_setup_push_tag_remove_tag(self):
        """Test push_tag and remove_tag on an OpenMLSetup object."""
        setup = openml.setups.get_setup(1)
        tag = "test_setup_tag_%d" % random.randint(1, 1_000_000)
        setup.push_tag(tag)
        setup.remove_tag(tag)
