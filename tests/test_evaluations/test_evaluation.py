# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.evaluations.evaluation module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openml.evaluations.evaluation import OpenMLEvaluation


class TestOpenMLEvaluationInit:
    """Test OpenMLEvaluation initialization."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        evaluation = OpenMLEvaluation(
            run_id=100,
            task_id=200,
            setup_id=300,
            flow_id=400,
            flow_name="sklearn.tree.DecisionTreeClassifier",
            data_id=500,
            data_name="iris",
            function="predictive_accuracy",
            upload_time="2023-01-15T10:30:00",
            uploader=600,
            uploader_name="test_user",
            value=0.95,
            values=[0.94, 0.95, 0.96],
            array_data="class_data_here"
        )
        
        assert evaluation.run_id == 100
        assert evaluation.task_id == 200
        assert evaluation.setup_id == 300
        assert evaluation.flow_id == 400
        assert evaluation.flow_name == "sklearn.tree.DecisionTreeClassifier"
        assert evaluation.data_id == 500
        assert evaluation.data_name == "iris"
        assert evaluation.function == "predictive_accuracy"
        assert evaluation.upload_time == "2023-01-15T10:30:00"
        assert evaluation.uploader == 600
        assert evaluation.uploader_name == "test_user"
        assert evaluation.value == 0.95
        assert evaluation.values == [0.94, 0.95, 0.96]
        assert evaluation.array_data == "class_data_here"

    def test_init_with_none_values(self):
        """Test initialization with None for optional parameters."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="test_flow",
            data_id=5,
            data_name="test_data",
            function="accuracy",
            upload_time="2023-01-01T00:00:00",
            uploader=6,
            uploader_name="user",
            value=None,
            values=None,
            array_data=None
        )
        
        assert evaluation.value is None
        assert evaluation.values is None
        assert evaluation.array_data is None

    def test_init_minimal_parameters(self):
        """Test initialization with minimal required parameters."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.5,
            values=[0.5]
        )
        
        assert evaluation.run_id == 1
        assert evaluation.array_data is None  # Default


class TestOpenMLEvaluationToDict:
    """Test _to_dict method."""

    def test_to_dict_complete(self):
        """Test _to_dict with all fields populated."""
        evaluation = OpenMLEvaluation(
            run_id=10,
            task_id=20,
            setup_id=30,
            flow_id=40,
            flow_name="test_flow",
            data_id=50,
            data_name="test_dataset",
            function="f1_score",
            upload_time="2023-06-15T14:30:00",
            uploader=60,
            uploader_name="john_doe",
            value=0.85,
            values=[0.84, 0.85, 0.86],
            array_data="array_info"
        )
        
        result = evaluation._to_dict()
        
        assert isinstance(result, dict)
        assert result["run_id"] == 10
        assert result["task_id"] == 20
        assert result["setup_id"] == 30
        assert result["flow_id"] == 40
        assert result["flow_name"] == "test_flow"
        assert result["data_id"] == 50
        assert result["data_name"] == "test_dataset"
        assert result["function"] == "f1_score"
        assert result["upload_time"] == "2023-06-15T14:30:00"
        assert result["uploader"] == 60
        assert result["uploader_name"] == "john_doe"
        assert result["value"] == 0.85
        assert result["values"] == [0.84, 0.85, 0.86]
        assert result["array_data"] == "array_info"

    def test_to_dict_with_none_values(self):
        """Test _to_dict when some values are None."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=None,
            values=None,
            array_data=None
        )
        
        result = evaluation._to_dict()
        
        assert result["value"] is None
        assert result["values"] is None
        assert result["array_data"] is None


class TestOpenMLEvaluationRepr:
    """Test __repr__ method."""

    def test_repr_format(self):
        """Test __repr__ output format."""
        evaluation = OpenMLEvaluation(
            run_id=123,
            task_id=456,
            setup_id=789,
            flow_id=111,
            flow_name="MyFlow",
            data_id=222,
            data_name="MyDataset",
            function="accuracy",
            upload_time="2023-01-01T12:00:00",
            uploader=333,
            uploader_name="testuser",
            value=0.92,
            values=[0.91, 0.92, 0.93]
        )
        
        repr_str = repr(evaluation)
        
        # Check header
        assert "OpenML Evaluation" in repr_str
        assert "=" in repr_str
        
        # Check key fields are present
        assert "123" in repr_str  # run_id
        assert "456" in repr_str  # task_id
        assert "111" in repr_str  # flow_id
        assert "222" in repr_str  # data_id
        assert "MyDataset" in repr_str
        assert "accuracy" in repr_str
        assert "0.92" in repr_str

    def test_repr_includes_urls(self):
        """Test that __repr__ includes OpenML URLs."""
        evaluation = OpenMLEvaluation(
            run_id=100,
            task_id=200,
            setup_id=300,
            flow_id=400,
            flow_name="flow",
            data_id=500,
            data_name="dataset",
            function="metric",
            upload_time="2023-01-01",
            uploader=600,
            uploader_name="user",
            value=0.8,
            values=[0.8]
        )
        
        with patch("openml.config.get_server_base_url", return_value="https://openml.org"):
            repr_str = repr(evaluation)
            
            # Should contain URL components
            assert "URL" in repr_str or "openml" in repr_str.lower()

    def test_repr_with_none_value(self):
        """Test __repr__ when value is None."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=None,
            values=None
        )
        
        repr_str = repr(evaluation)
        
        # Should still be valid output
        assert "OpenML Evaluation" in repr_str


class TestOpenMLEvaluationAttributes:
    """Test attribute access and manipulation."""

    def test_all_attributes_accessible(self):
        """Test that all attributes are accessible."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.5,
            values=[0.5]
        )
        
        # All attributes should be accessible
        assert hasattr(evaluation, "run_id")
        assert hasattr(evaluation, "task_id")
        assert hasattr(evaluation, "setup_id")
        assert hasattr(evaluation, "flow_id")
        assert hasattr(evaluation, "flow_name")
        assert hasattr(evaluation, "data_id")
        assert hasattr(evaluation, "data_name")
        assert hasattr(evaluation, "function")
        assert hasattr(evaluation, "upload_time")
        assert hasattr(evaluation, "uploader")
        assert hasattr(evaluation, "uploader_name")
        assert hasattr(evaluation, "value")
        assert hasattr(evaluation, "values")
        assert hasattr(evaluation, "array_data")

    def test_attribute_modification(self):
        """Test that attributes can be modified after initialization."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.5,
            values=[0.5]
        )
        
        # Modify attributes
        evaluation.value = 0.75
        evaluation.flow_name = "new_flow"
        
        assert evaluation.value == 0.75
        assert evaluation.flow_name == "new_flow"


class TestOpenMLEvaluationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_value(self):
        """Test evaluation with zero value."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.0,
            values=[0.0, 0.0, 0.0]
        )
        
        assert evaluation.value == 0.0
        assert all(v == 0.0 for v in evaluation.values)

    def test_negative_value(self):
        """Test evaluation with negative value."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=-0.5,
            values=[-0.4, -0.5, -0.6]
        )
        
        assert evaluation.value == -0.5

    def test_very_large_value(self):
        """Test evaluation with very large value."""
        large_value = 1e10
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=large_value,
            values=[large_value]
        )
        
        assert evaluation.value == large_value

    def test_empty_values_list(self):
        """Test evaluation with empty values list."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.5,
            values=[]
        )
        
        assert evaluation.values == []

    def test_many_values(self):
        """Test evaluation with many values (e.g., many folds)."""
        many_values = [0.8 + i * 0.01 for i in range(100)]
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.85,
            values=many_values
        )
        
        assert len(evaluation.values) == 100

    def test_special_characters_in_names(self):
        """Test with special characters in names."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="sklearn.ensemble.RandomForestClassifier(n_estimators=100)",
            data_id=5,
            data_name="dataset-with-dashes_and_underscores",
            function="f1_weighted",
            upload_time="2023-01-01T12:34:56",
            uploader=6,
            uploader_name="user_name_123",
            value=0.9,
            values=[0.9]
        )
        
        assert "(" in evaluation.flow_name
        assert "-" in evaluation.data_name
        assert "_" in evaluation.uploader_name

    def test_unicode_in_names(self):
        """Test with unicode characters in names."""
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="分類器",
            data_id=5,
            data_name="データセット",
            function="精度",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="ユーザー",
            value=0.9,
            values=[0.9]
        )
        
        assert "分類器" in evaluation.flow_name
        assert "データセット" in evaluation.data_name

    def test_very_long_array_data(self):
        """Test with very long array_data string."""
        long_array_data = "x" * 100000
        evaluation = OpenMLEvaluation(
            run_id=1,
            task_id=2,
            setup_id=3,
            flow_id=4,
            flow_name="flow",
            data_id=5,
            data_name="data",
            function="metric",
            upload_time="2023-01-01",
            uploader=6,
            uploader_name="user",
            value=0.5,
            values=[0.5],
            array_data=long_array_data
        )
        
        assert len(evaluation.array_data) == 100000

    def test_timestamp_formats(self):
        """Test with different timestamp formats."""
        timestamps = [
            "2023-01-01T00:00:00",
            "2023-12-31T23:59:59",
            "2023-06-15T12:30:45.123456",
            "2023-01-01",
        ]
        
        for timestamp in timestamps:
            evaluation = OpenMLEvaluation(
                run_id=1,
                task_id=2,
                setup_id=3,
                flow_id=4,
                flow_name="flow",
                data_id=5,
                data_name="data",
                function="metric",
                upload_time=timestamp,
                uploader=6,
                uploader_name="user",
                value=0.5,
                values=[0.5]
            )
            
            assert evaluation.upload_time == timestamp


class TestOpenMLEvaluationIntegration:
    """Integration tests for OpenMLEvaluation."""

    def test_create_multiple_evaluations(self):
        """Test creating multiple evaluation objects."""
        evaluations = []
        
        for i in range(10):
            eval_obj = OpenMLEvaluation(
                run_id=i * 10,
                task_id=i * 20,
                setup_id=i * 30,
                flow_id=i * 40,
                flow_name=f"flow_{i}",
                data_id=i * 50,
                data_name=f"data_{i}",
                function="accuracy",
                upload_time=f"2023-01-{i+1:02d}",
                uploader=i * 60,
                uploader_name=f"user_{i}",
                value=0.8 + i * 0.01,
                values=[0.8 + i * 0.01]
            )
            evaluations.append(eval_obj)
        
        assert len(evaluations) == 10
        assert all(isinstance(e, OpenMLEvaluation) for e in evaluations)

    def test_evaluation_to_dict_roundtrip(self):
        """Test that _to_dict preserves all information."""
        original = OpenMLEvaluation(
            run_id=100,
            task_id=200,
            setup_id=300,
            flow_id=400,
            flow_name="test_flow",
            data_id=500,
            data_name="test_data",
            function="f1",
            upload_time="2023-01-01",
            uploader=600,
            uploader_name="user",
            value=0.88,
            values=[0.87, 0.88, 0.89],
            array_data="test_array"
        )
        
        dict_repr = original._to_dict()
        
        # Create new evaluation from dict
        reconstructed = OpenMLEvaluation(**dict_repr)
        
        # Check all fields match
        assert reconstructed.run_id == original.run_id
        assert reconstructed.task_id == original.task_id
        assert reconstructed.setup_id == original.setup_id
        assert reconstructed.flow_id == original.flow_id
        assert reconstructed.flow_name == original.flow_name
        assert reconstructed.data_id == original.data_id
        assert reconstructed.data_name == original.data_name
        assert reconstructed.function == original.function
        assert reconstructed.value == original.value

    def test_different_metrics(self):
        """Test evaluations with different metric types."""
        metrics = [
            "predictive_accuracy",
            "area_under_roc_curve",
            "f_measure",
            "precision",
            "recall",
            "mean_absolute_error",
            "root_mean_squared_error"
        ]
        
        for metric in metrics:
            evaluation = OpenMLEvaluation(
                run_id=1,
                task_id=2,
                setup_id=3,
                flow_id=4,
                flow_name="flow",
                data_id=5,
                data_name="data",
                function=metric,
                upload_time="2023-01-01",
                uploader=6,
                uploader_name="user",
                value=0.5,
                values=[0.5]
            )
            
            assert evaluation.function == metric
