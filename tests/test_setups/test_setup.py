# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.setups.setup module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openml.setups.setup import OpenMLParameter, OpenMLSetup


class TestOpenMLSetupInit:
    """Test OpenMLSetup initialization."""

    def test_init_with_parameters(self):
        """Test initialization with parameters."""
        param1 = OpenMLParameter(
            input_id=1,
            flow_id=100,
            flow_name="flow1",
            full_name="flow1.param1",
            parameter_name="param1",
            data_type="int",
            default_value="10",
            value="20"
        )
        
        parameters = {1: param1}
        
        setup = OpenMLSetup(
            setup_id=500,
            flow_id=100,
            parameters=parameters
        )
        
        assert setup.setup_id == 500
        assert setup.flow_id == 100
        assert setup.parameters == parameters
        assert 1 in setup.parameters

    def test_init_without_parameters(self):
        """Test initialization without parameters."""
        setup = OpenMLSetup(
            setup_id=123,
            flow_id=456,
            parameters=None
        )
        
        assert setup.setup_id == 123
        assert setup.flow_id == 456
        assert setup.parameters is None

    def test_invalid_setup_id_type(self):
        """Test that non-int setup_id raises ValueError."""
        with pytest.raises(ValueError, match="setup id should be int"):
            OpenMLSetup(
                setup_id="123",  # String instead of int
                flow_id=456,
                parameters=None
            )

    def test_invalid_flow_id_type(self):
        """Test that non-int flow_id raises ValueError."""
        with pytest.raises(ValueError, match="flow id should be int"):
            OpenMLSetup(
                setup_id=123,
                flow_id="456",  # String instead of int
                parameters=None
            )

    def test_invalid_parameters_type(self):
        """Test that non-dict parameters raises ValueError."""
        with pytest.raises(ValueError, match="parameters should be dict"):
            OpenMLSetup(
                setup_id=123,
                flow_id=456,
                parameters="not a dict"
            )


class TestOpenMLSetupToDict:
    """Test _to_dict method."""

    def test_to_dict_with_parameters(self):
        """Test _to_dict with parameters."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=100,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="str",
            default_value="default",
            value="custom"
        )
        
        parameters = {1: param}
        setup = OpenMLSetup(
            setup_id=200,
            flow_id=100,
            parameters=parameters
        )
        
        result = setup._to_dict()
        
        assert result["setup_id"] == 200
        assert result["flow_id"] == 100
        assert result["parameters"] is not None
        assert 1 in result["parameters"]

    def test_to_dict_without_parameters(self):
        """Test _to_dict without parameters."""
        setup = OpenMLSetup(
            setup_id=300,
            flow_id=400,
            parameters=None
        )
        
        result = setup._to_dict()
        
        assert result["setup_id"] == 300
        assert result["flow_id"] == 400
        assert result["parameters"] is None


class TestOpenMLSetupRepr:
    """Test __repr__ method."""

    def test_repr_with_parameters(self):
        """Test __repr__ with parameters."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=100,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="int",
            default_value="5",
            value="10"
        )
        
        setup = OpenMLSetup(
            setup_id=123,
            flow_id=100,
            parameters={1: param}
        )
        
        repr_str = repr(setup)
        
        assert "OpenML Setup" in repr_str
        assert "123" in repr_str  # setup_id
        assert "100" in repr_str  # flow_id
        assert "1" in repr_str  # number of parameters

    def test_repr_without_parameters(self):
        """Test __repr__ without parameters."""
        setup = OpenMLSetup(
            setup_id=456,
            flow_id=789,
            parameters=None
        )
        
        repr_str = repr(setup)
        
        assert "OpenML Setup" in repr_str
        assert "456" in repr_str
        assert "789" in repr_str
        assert "nan" in repr_str  # number of parameters when None

    def test_repr_includes_flow_url(self):
        """Test that __repr__ includes flow URL."""
        setup = OpenMLSetup(
            setup_id=1,
            flow_id=2,
            parameters=None
        )
        
        with patch("openml.config.get_server_base_url", return_value="https://openml.org"):
            repr_str = repr(setup)
            
            # URL should be mentioned
            assert "URL" in repr_str or "openml.org" in repr_str


class TestOpenMLParameterInit:
    """Test OpenMLParameter initialization."""

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        param = OpenMLParameter(
            input_id=10,
            flow_id=20,
            flow_name="sklearn.tree.DecisionTreeClassifier",
            full_name="sklearn.tree.DecisionTreeClassifier.max_depth",
            parameter_name="max_depth",
            data_type="int",
            default_value="None",
            value="10"
        )
        
        assert param.id == 10
        assert param.flow_id == 20
        assert param.flow_name == "sklearn.tree.DecisionTreeClassifier"
        assert param.full_name == "sklearn.tree.DecisionTreeClassifier.max_depth"
        assert param.parameter_name == "max_depth"
        assert param.data_type == "int"
        assert param.default_value == "None"
        assert param.value == "10"

    def test_init_minimal_parameters(self):
        """Test initialization with minimal parameters."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="str",
            default_value="default",
            value="value"
        )
        
        assert param.id == 1
        assert param.parameter_name == "param"


class TestOpenMLParameterToDict:
    """Test _to_dict method of OpenMLParameter."""

    def test_to_dict(self):
        """Test _to_dict returns correct dictionary."""
        param = OpenMLParameter(
            input_id=100,
            flow_id=200,
            flow_name="test_flow",
            full_name="test_flow.test_param",
            parameter_name="test_param",
            data_type="float",
            default_value="0.5",
            value="0.7"
        )
        
        result = param._to_dict()
        
        assert result["id"] == 100
        assert result["flow_id"] == 200
        assert result["flow_name"] == "test_flow"
        assert result["full_name"] == "test_flow.test_param"
        assert result["parameter_name"] == "test_param"
        assert result["data_type"] == "float"
        assert result["default_value"] == "0.5"
        assert result["value"] == "0.7"


class TestOpenMLParameterRepr:
    """Test __repr__ method of OpenMLParameter."""

    def test_repr_format(self):
        """Test __repr__ output format."""
        param = OpenMLParameter(
            input_id=50,
            flow_id=100,
            flow_name="MyFlow",
            full_name="MyFlow.MyParam",
            parameter_name="MyParam",
            data_type="int",
            default_value="5",
            value="10"
        )
        
        repr_str = repr(param)
        
        assert "OpenML Parameter" in repr_str
        assert "50" in repr_str  # ID
        assert "100" in repr_str  # Flow ID
        assert "MyParam" in repr_str
        assert "int" in repr_str
        assert "5" in repr_str  # default
        assert "10" in repr_str  # value

    def test_repr_includes_flow_url(self):
        """Test that __repr__ includes flow URL."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="str",
            default_value="def",
            value="val"
        )
        
        with patch("openml.config.get_server_base_url", return_value="https://openml.org"):
            repr_str = repr(param)
            
            assert "URL" in repr_str or "openml.org" in repr_str

    def test_repr_with_indentation(self):
        """Test that __repr__ shows indented parameter attributes."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="bool",
            default_value="True",
            value="False"
        )
        
        repr_str = repr(param)
        
        # Should contain indented attributes
        assert "|__" in repr_str or "|_" in repr_str


class TestOpenMLSetupEdgeCases:
    """Test edge cases for OpenMLSetup."""

    def test_setup_with_many_parameters(self):
        """Test setup with many parameters."""
        parameters = {}
        for i in range(100):
            param = OpenMLParameter(
                input_id=i,
                flow_id=1,
                flow_name="flow",
                full_name=f"flow.param{i}",
                parameter_name=f"param{i}",
                data_type="int",
                default_value=str(i),
                value=str(i * 2)
            )
            parameters[i] = param
        
        setup = OpenMLSetup(
            setup_id=1,
            flow_id=1,
            parameters=parameters
        )
        
        assert len(setup.parameters) == 100

    def test_setup_with_empty_parameters_dict(self):
        """Test setup with empty parameters dictionary."""
        setup = OpenMLSetup(
            setup_id=1,
            flow_id=2,
            parameters={}
        )
        
        assert setup.parameters == {}
        assert len(setup.parameters) == 0

    def test_setup_zero_ids(self):
        """Test setup with zero IDs."""
        setup = OpenMLSetup(
            setup_id=0,
            flow_id=0,
            parameters=None
        )
        
        assert setup.setup_id == 0
        assert setup.flow_id == 0

    def test_setup_negative_ids(self):
        """Test that negative IDs are accepted (implementation allows it)."""
        # Note: While unusual, the implementation doesn't explicitly forbid negative IDs
        setup = OpenMLSetup(
            setup_id=-1,
            flow_id=-2,
            parameters=None
        )
        
        assert setup.setup_id == -1
        assert setup.flow_id == -2


class TestOpenMLParameterEdgeCases:
    """Test edge cases for OpenMLParameter."""

    def test_parameter_with_none_as_value(self):
        """Test parameter with 'None' as string value."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="object",
            default_value="None",
            value="None"
        )
        
        assert param.value == "None"
        assert param.default_value == "None"

    def test_parameter_with_empty_strings(self):
        """Test parameter with empty strings."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="",
            full_name="",
            parameter_name="",
            data_type="",
            default_value="",
            value=""
        )
        
        assert param.flow_name == ""
        assert param.parameter_name == ""

    def test_parameter_with_special_characters(self):
        """Test parameter with special characters in names."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="sklearn.ensemble.RandomForestClassifier",
            full_name="sklearn.ensemble.RandomForestClassifier.max_features",
            parameter_name="max_features",
            data_type="str",
            default_value="auto",
            value="sqrt"
        )
        
        assert "." in param.flow_name
        assert "_" in param.parameter_name

    def test_parameter_with_complex_values(self):
        """Test parameter with complex string values."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="tuple",
            default_value="(1, 2, 3)",
            value="(10, 20, 30)"
        )
        
        assert param.value == "(10, 20, 30)"

    def test_parameter_with_unicode(self):
        """Test parameter with unicode characters."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=2,
            flow_name="フロー",
            full_name="フロー.パラメータ",
            parameter_name="パラメータ",
            data_type="文字列",
            default_value="デフォルト",
            value="値"
        )
        
        assert "フロー" in param.flow_name


class TestOpenMLSetupIntegration:
    """Integration tests for OpenMLSetup and OpenMLParameter."""

    def test_setup_with_multiple_parameters(self):
        """Test setup with multiple different parameters."""
        param1 = OpenMLParameter(
            input_id=1,
            flow_id=100,
            flow_name="flow",
            full_name="flow.param1",
            parameter_name="param1",
            data_type="int",
            default_value="5",
            value="10"
        )
        
        param2 = OpenMLParameter(
            input_id=2,
            flow_id=100,
            flow_name="flow",
            full_name="flow.param2",
            parameter_name="param2",
            data_type="float",
            default_value="0.5",
            value="0.8"
        )
        
        param3 = OpenMLParameter(
            input_id=3,
            flow_id=100,
            flow_name="flow",
            full_name="flow.param3",
            parameter_name="param3",
            data_type="str",
            default_value="default",
            value="custom"
        )
        
        parameters = {1: param1, 2: param2, 3: param3}
        setup = OpenMLSetup(
            setup_id=500,
            flow_id=100,
            parameters=parameters
        )
        
        assert len(setup.parameters) == 3
        assert all(isinstance(p, OpenMLParameter) for p in setup.parameters.values())

    def test_setup_to_dict_with_parameters_calls_param_to_dict(self):
        """Test that setup._to_dict calls _to_dict on parameters."""
        param = OpenMLParameter(
            input_id=1,
            flow_id=100,
            flow_name="flow",
            full_name="flow.param",
            parameter_name="param",
            data_type="int",
            default_value="5",
            value="10"
        )
        
        setup = OpenMLSetup(
            setup_id=200,
            flow_id=100,
            parameters={1: param}
        )
        
        result = setup._to_dict()
        
        # Check that parameter dict is included
        assert "parameters" in result
        assert 1 in result["parameters"]
        assert isinstance(result["parameters"][1], dict)

    def test_parameter_different_data_types(self):
        """Test parameters with different data types."""
        data_types = ["int", "float", "str", "bool", "list", "dict", "tuple", "object"]
        
        for i, dtype in enumerate(data_types):
            param = OpenMLParameter(
                input_id=i,
                flow_id=1,
                flow_name="flow",
                full_name=f"flow.param{i}",
                parameter_name=f"param{i}",
                data_type=dtype,
                default_value="default",
                value="value"
            )
            
            assert param.data_type == dtype

    def test_setup_repr_with_multiple_parameters(self):
        """Test setup __repr__ correctly shows parameter count."""
        parameters = {}
        for i in range(5):
            param = OpenMLParameter(
                input_id=i,
                flow_id=1,
                flow_name="flow",
                full_name=f"flow.p{i}",
                parameter_name=f"p{i}",
                data_type="int",
                default_value="0",
                value="1"
            )
            parameters[i] = param
        
        setup = OpenMLSetup(
            setup_id=1,
            flow_id=1,
            parameters=parameters
        )
        
        repr_str = repr(setup)
        
        assert "5" in repr_str  # Should show 5 parameters
