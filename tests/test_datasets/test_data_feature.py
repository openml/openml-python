# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.datasets.data_feature module."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from openml.datasets.data_feature import OpenMLDataFeature


class TestOpenMLDataFeatureInit:
    """Test OpenMLDataFeature initialization."""

    def test_init_nominal_feature(self):
        """Test initialization of nominal feature."""
        feature = OpenMLDataFeature(
            index=0,
            name="color",
            data_type="nominal",
            nominal_values=["red", "blue", "green"],
            number_missing_values=5
        )
        
        assert feature.index == 0
        assert feature.name == "color"
        assert feature.data_type == "nominal"
        assert feature.nominal_values == ["red", "blue", "green"]
        assert feature.number_missing_values == 5
        assert feature.ontologies is None

    def test_init_numeric_feature(self):
        """Test initialization of numeric feature."""
        feature = OpenMLDataFeature(
            index=1,
            name="age",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature.index == 1
        assert feature.name == "age"
        assert feature.data_type == "numeric"
        assert feature.nominal_values is None
        assert feature.number_missing_values == 0

    def test_init_string_feature(self):
        """Test initialization of string feature."""
        feature = OpenMLDataFeature(
            index=2,
            name="description",
            data_type="string",
            nominal_values=None,
            number_missing_values=10
        )
        
        assert feature.data_type == "string"
        assert feature.nominal_values is None

    def test_init_date_feature(self):
        """Test initialization of date feature."""
        feature = OpenMLDataFeature(
            index=3,
            name="timestamp",
            data_type="date",
            nominal_values=None,
            number_missing_values=2
        )
        
        assert feature.data_type == "date"
        assert feature.nominal_values is None

    def test_init_with_ontologies(self):
        """Test initialization with ontologies."""
        ontologies = ["http://example.org/ontology1", "http://example.org/ontology2"]
        feature = OpenMLDataFeature(
            index=0,
            name="feature",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=ontologies
        )
        
        assert feature.ontologies == ontologies

    def test_init_name_conversion_to_string(self):
        """Test that name is converted to string."""
        feature = OpenMLDataFeature(
            index=0,
            name=12345,  # Integer name
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature.name == "12345"
        assert isinstance(feature.name, str)

    def test_init_data_type_conversion_to_string(self):
        """Test that data_type is converted to string."""
        feature = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert isinstance(feature.data_type, str)


class TestOpenMLDataFeatureValidation:
    """Test validation in OpenMLDataFeature."""

    def test_invalid_index_type(self):
        """Test that non-integer index raises TypeError."""
        with pytest.raises(TypeError, match="Index must be `int`"):
            OpenMLDataFeature(
                index="0",  # String instead of int
                name="feature",
                data_type="numeric",
                nominal_values=None,
                number_missing_values=0
            )

    def test_invalid_data_type(self):
        """Test that invalid data_type raises ValueError."""
        with pytest.raises(ValueError, match="data type should be in"):
            OpenMLDataFeature(
                index=0,
                name="feature",
                data_type="invalid_type",
                nominal_values=None,
                number_missing_values=0
            )

    def test_nominal_without_nominal_values(self):
        """Test that nominal type without values raises TypeError."""
        with pytest.raises(TypeError, match="require attribute `nominal_values`"):
            OpenMLDataFeature(
                index=0,
                name="color",
                data_type="nominal",
                nominal_values=None,
                number_missing_values=0
            )

    def test_nominal_values_not_list(self):
        """Test that non-list nominal_values raises TypeError."""
        with pytest.raises(TypeError, match="should be list"):
            OpenMLDataFeature(
                index=0,
                name="color",
                data_type="nominal",
                nominal_values="red,blue,green",  # String instead of list
                number_missing_values=0
            )

    def test_numeric_with_nominal_values(self):
        """Test that numeric type with nominal_values raises TypeError."""
        with pytest.raises(TypeError, match="must be None for non-nominal"):
            OpenMLDataFeature(
                index=0,
                name="age",
                data_type="numeric",
                nominal_values=["1", "2", "3"],
                number_missing_values=0
            )

    def test_invalid_number_missing_values_type(self):
        """Test that non-int number_missing_values raises TypeError."""
        with pytest.raises(TypeError, match="number_missing_values must be int"):
            OpenMLDataFeature(
                index=0,
                name="feature",
                data_type="numeric",
                nominal_values=None,
                number_missing_values="5"  # String instead of int
            )


class TestOpenMLDataFeatureLegalDataTypes:
    """Test LEGAL_DATA_TYPES class variable."""

    def test_legal_data_types_exists(self):
        """Test that LEGAL_DATA_TYPES class variable exists."""
        assert hasattr(OpenMLDataFeature, "LEGAL_DATA_TYPES")

    def test_legal_data_types_content(self):
        """Test that LEGAL_DATA_TYPES contains expected types."""
        legal_types = OpenMLDataFeature.LEGAL_DATA_TYPES
        
        assert "nominal" in legal_types
        assert "numeric" in legal_types
        assert "string" in legal_types
        assert "date" in legal_types

    @pytest.mark.parametrize("data_type", ["nominal", "numeric", "string", "date"])
    def test_all_legal_types_accepted(self, data_type):
        """Test that all legal data types are accepted."""
        nominal_values = ["a", "b"] if data_type == "nominal" else None
        
        feature = OpenMLDataFeature(
            index=0,
            name="test",
            data_type=data_type,
            nominal_values=nominal_values,
            number_missing_values=0
        )
        
        assert feature.data_type == data_type


class TestOpenMLDataFeatureRepr:
    """Test __repr__ method."""

    def test_repr_format(self):
        """Test __repr__ returns expected format."""
        feature = OpenMLDataFeature(
            index=5,
            name="test_feature",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        repr_str = repr(feature)
        
        assert "[5 - test_feature (numeric)]" == repr_str

    def test_repr_nominal_feature(self):
        """Test __repr__ for nominal feature."""
        feature = OpenMLDataFeature(
            index=0,
            name="category",
            data_type="nominal",
            nominal_values=["A", "B", "C"],
            number_missing_values=0
        )
        
        repr_str = repr(feature)
        
        assert "[0 - category (nominal)]" == repr_str
        assert "0" in repr_str
        assert "category" in repr_str
        assert "nominal" in repr_str

    def test_repr_with_special_characters(self):
        """Test __repr__ with special characters in name."""
        feature = OpenMLDataFeature(
            index=10,
            name="feature_with-special.chars",
            data_type="string",
            nominal_values=None,
            number_missing_values=0
        )
        
        repr_str = repr(feature)
        
        assert "feature_with-special.chars" in repr_str


class TestOpenMLDataFeatureEquality:
    """Test __eq__ method."""

    def test_equality_same_features(self):
        """Test that identical features are equal."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=5
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=5
        )
        
        assert feature1 == feature2

    def test_equality_different_index(self):
        """Test that features with different index are not equal."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        feature2 = OpenMLDataFeature(
            index=1,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature1 != feature2

    def test_equality_different_name(self):
        """Test that features with different name are not equal."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="feature1",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="feature2",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature1 != feature2

    def test_equality_different_data_type(self):
        """Test that features with different data_type are not equal."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="string",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature1 != feature2

    def test_equality_different_nominal_values(self):
        """Test that features with different nominal_values are not equal."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="nominal",
            nominal_values=["A", "B"],
            number_missing_values=0
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="nominal",
            nominal_values=["A", "C"],
            number_missing_values=0
        )
        
        assert feature1 != feature2

    def test_equality_different_missing_values(self):
        """Test that features with different number_missing_values are not equal."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=5
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=10
        )
        
        assert feature1 != feature2

    def test_equality_with_ontologies(self):
        """Test equality with ontologies."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=["onto1", "onto2"]
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=["onto1", "onto2"]
        )
        
        assert feature1 == feature2

    def test_equality_different_ontologies(self):
        """Test inequality with different ontologies."""
        feature1 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=["onto1"]
        )
        
        feature2 = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=["onto2"]
        )
        
        assert feature1 != feature2

    def test_equality_with_non_feature_object(self):
        """Test that feature is not equal to non-feature object."""
        feature = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature != "not a feature"
        assert feature != 123
        assert feature != None
        assert feature != {"index": 0}


class TestOpenMLDataFeaturePrettyRepr:
    """Test _repr_pretty_ method for IPython."""

    def test_repr_pretty_exists(self):
        """Test that _repr_pretty_ method exists."""
        feature = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert hasattr(feature, "_repr_pretty_")

    def test_repr_pretty_output(self):
        """Test _repr_pretty_ produces correct output."""
        feature = OpenMLDataFeature(
            index=3,
            name="my_feature",
            data_type="nominal",
            nominal_values=["X", "Y", "Z"],
            number_missing_values=7
        )
        
        # Mock pretty printer
        mock_pp = Mock()
        feature._repr_pretty_(mock_pp, False)
        
        # Should call text with the string representation
        mock_pp.text.assert_called_once()
        call_arg = mock_pp.text.call_args[0][0]
        assert "[3 - my_feature (nominal)]" == call_arg


class TestOpenMLDataFeatureEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_missing_values(self):
        """Test feature with zero missing values."""
        feature = OpenMLDataFeature(
            index=0,
            name="complete",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature.number_missing_values == 0

    def test_large_number_missing_values(self):
        """Test feature with large number of missing values."""
        feature = OpenMLDataFeature(
            index=0,
            name="sparse",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=999999
        )
        
        assert feature.number_missing_values == 999999

    def test_negative_index(self):
        """Test that negative index is accepted (Python allows negative int)."""
        feature = OpenMLDataFeature(
            index=-1,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature.index == -1

    def test_empty_nominal_values_list(self):
        """Test nominal feature with empty nominal_values list."""
        feature = OpenMLDataFeature(
            index=0,
            name="empty_nominal",
            data_type="nominal",
            nominal_values=[],
            number_missing_values=0
        )
        
        assert feature.nominal_values == []

    def test_single_nominal_value(self):
        """Test nominal feature with single value."""
        feature = OpenMLDataFeature(
            index=0,
            name="single_value",
            data_type="nominal",
            nominal_values=["only_value"],
            number_missing_values=0
        )
        
        assert feature.nominal_values == ["only_value"]

    def test_many_nominal_values(self):
        """Test nominal feature with many values."""
        many_values = [f"value_{i}" for i in range(1000)]
        feature = OpenMLDataFeature(
            index=0,
            name="many_values",
            data_type="nominal",
            nominal_values=many_values,
            number_missing_values=0
        )
        
        assert len(feature.nominal_values) == 1000

    def test_unicode_in_name(self):
        """Test feature with unicode characters in name."""
        feature = OpenMLDataFeature(
            index=0,
            name="特徴_データ",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert "特徴" in feature.name

    def test_unicode_in_nominal_values(self):
        """Test nominal feature with unicode values."""
        feature = OpenMLDataFeature(
            index=0,
            name="language",
            data_type="nominal",
            nominal_values=["English", "日本語", "中文", "Español"],
            number_missing_values=0
        )
        
        assert "日本語" in feature.nominal_values

    def test_empty_name(self):
        """Test feature with empty name."""
        feature = OpenMLDataFeature(
            index=0,
            name="",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert feature.name == ""

    def test_very_long_name(self):
        """Test feature with very long name."""
        long_name = "x" * 10000
        feature = OpenMLDataFeature(
            index=0,
            name=long_name,
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0
        )
        
        assert len(feature.name) == 10000

    def test_multiple_ontologies(self):
        """Test feature with multiple ontologies."""
        ontologies = [f"http://onto{i}.org" for i in range(10)]
        feature = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=ontologies
        )
        
        assert len(feature.ontologies) == 10

    def test_empty_ontologies_list(self):
        """Test feature with empty ontologies list."""
        feature = OpenMLDataFeature(
            index=0,
            name="test",
            data_type="numeric",
            nominal_values=None,
            number_missing_values=0,
            ontologies=[]
        )
        
        assert feature.ontologies == []


class TestOpenMLDataFeatureIntegration:
    """Integration tests for OpenMLDataFeature."""

    def test_create_multiple_features(self):
        """Test creating multiple features for a dataset."""
        features = [
            OpenMLDataFeature(0, "id", "numeric", None, 0),
            OpenMLDataFeature(1, "name", "string", None, 5),
            OpenMLDataFeature(2, "category", "nominal", ["A", "B", "C"], 2),
            OpenMLDataFeature(3, "value", "numeric", None, 10),
            OpenMLDataFeature(4, "date", "date", None, 0),
        ]
        
        assert len(features) == 5
        assert all(isinstance(f, OpenMLDataFeature) for f in features)

    def test_feature_comparison_in_list(self):
        """Test feature comparison when stored in list."""
        feature1 = OpenMLDataFeature(0, "test", "numeric", None, 0)
        feature2 = OpenMLDataFeature(0, "test", "numeric", None, 0)
        feature3 = OpenMLDataFeature(1, "other", "numeric", None, 0)
        
        features = [feature1, feature3]
        
        assert feature2 in [feature1]  # Equal to feature1
        assert feature1 == feature2
        assert feature3 not in [feature1]
