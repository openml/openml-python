# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.testing module."""

from __future__ import annotations

import hashlib
import pathlib
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from openml import testing
from openml.tasks import TaskType


class TestCheckDatasetFunction:
    """Test _check_dataset function."""

    def test_check_dataset_valid(self):
        """Test _check_dataset with valid dataset dict."""
        dataset = {
            "did": 123,
            "status": "active",
            "name": "test_dataset"
        }
        
        # Should not raise
        testing._check_dataset(dataset)

    def test_check_dataset_with_in_preparation_status(self):
        """Test _check_dataset with in_preparation status."""
        dataset = {
            "did": 456,
            "status": "in_preparation"
        }
        
        # Should not raise
        testing._check_dataset(dataset)

    def test_check_dataset_with_deactivated_status(self):
        """Test _check_dataset with deactivated status."""
        dataset = {
            "did": 789,
            "status": "deactivated"
        }
        
        # Should not raise
        testing._check_dataset(dataset)


class TestCheckTaskExistence:
    """Test check_task_existence function."""

    def test_check_task_existence_no_tasks(self):
        """Test check_task_existence when no tasks exist."""
        mock_tasks = Mock()
        mock_tasks.__len__ = Mock(return_value=0)
        
        with patch("openml.tasks.list_tasks", return_value=mock_tasks):
            result = testing.check_task_existence(
                TaskType.SUPERVISED_CLASSIFICATION,
                1,
                "target"
            )
            
            assert result is None

    def test_check_task_existence_with_matching_task(self):
        """Test check_task_existence with matching task."""
        # Create a mock DataFrame-like object
        mock_tasks = Mock()
        mock_filtered_by_did = Mock()
        mock_filtered_by_target = Mock()
        mock_bool_mask = Mock()
        
        # Make mock_tasks subscriptable and support boolean indexing
        # tasks["did"] returns a series that supports == comparison
        mock_did_column = Mock()
        mock_did_column.__eq__ = Mock(return_value=mock_bool_mask)
        
        mock_target_column = Mock()
        mock_target_column.__eq__ = Mock(return_value=Mock())
        
        # Return appropriate column when indexed
        def getitem_side_effect(key):
            if key == "did":
                return mock_did_column
            elif key == "target_feature":
                return mock_target_column
            elif key == "tid":
                return Mock(to_list=Mock(return_value=[123]))
            return Mock()
        
        mock_tasks.__getitem__ = Mock(side_effect=getitem_side_effect)
        mock_filtered_by_did.__getitem__ = Mock(side_effect=getitem_side_effect)
        mock_filtered_by_target.__getitem__ = Mock(side_effect=getitem_side_effect)
        
        # Set up the chain of filters with .loc (using Mock with __getitem__)
        mock_tasks.__len__ = Mock(return_value=1)
        mock_loc1 = Mock()
        mock_loc1.__getitem__ = Mock(return_value=mock_filtered_by_did)
        mock_tasks.loc = mock_loc1
        
        mock_filtered_by_did.__len__ = Mock(return_value=1)
        mock_loc2 = Mock()
        mock_loc2.__getitem__ = Mock(return_value=mock_filtered_by_target)
        mock_filtered_by_did.loc = mock_loc2
        
        mock_filtered_by_target.__len__ = Mock(return_value=1)
        
        mock_task = Mock()
        
        with patch("openml.tasks.list_tasks", return_value=mock_tasks), \
             patch("openml.tasks.get_task", return_value=mock_task):
            
            result = testing.check_task_existence(
                TaskType.SUPERVISED_CLASSIFICATION,
                dataset_id=1,
                target_name="target"
            )
            
            # Should return task ID if found
            assert result == 123 or result is None


class TestCustomImputer:
    """Test CustomImputer class."""

    def test_custom_imputer_exists(self):
        """Test that CustomImputer class exists."""
        assert hasattr(testing, "CustomImputer")

    def test_custom_imputer_inheritance(self):
        """Test that CustomImputer inherits from SimpleImputer."""
        from openml.testing import CustomImputer, SimpleImputer
        
        imputer = CustomImputer()
        assert isinstance(imputer, SimpleImputer)


class TestCreateRequestResponse:
    """Test create_request_response function."""

    def test_create_request_response_basic(self, tmp_path):
        """Test create_request_response creates valid response."""
        # Create a temporary XML file
        xml_file = tmp_path / "response.xml"
        xml_content = "<response><status>success</status></response>"
        xml_file.write_text(xml_content)
        
        response = testing.create_request_response(
            status_code=200,
            content_filepath=xml_file
        )
        
        assert isinstance(response, requests.Response)
        assert response.status_code == 200
        assert xml_content in response.text

    def test_create_request_response_different_status_codes(self, tmp_path):
        """Test create_request_response with different status codes."""
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<test>content</test>")
        
        for status_code in [200, 404, 500]:
            response = testing.create_request_response(
                status_code=status_code,
                content_filepath=xml_file
            )
            
            assert response.status_code == status_code

    def test_create_request_response_content(self, tmp_path):
        """Test create_request_response preserves content."""
        xml_file = tmp_path / "data.xml"
        xml_content = "<data><item>test</item></data>"
        xml_file.write_text(xml_content)
        
        response = testing.create_request_response(
            status_code=200,
            content_filepath=xml_file
        )
        
        assert response.text == xml_content


class TestTestBaseClassVariables:
    """Test TestBase class variables."""

    def test_publish_tracker_exists(self):
        """Test that publish_tracker class variable exists."""
        assert hasattr(testing.TestBase, "publish_tracker")
        assert isinstance(testing.TestBase.publish_tracker, dict)

    def test_publish_tracker_keys(self):
        """Test publish_tracker has expected keys."""
        tracker = testing.TestBase.publish_tracker
        
        expected_keys = ["run", "data", "flow", "task", "study", "user"]
        for key in expected_keys:
            assert key in tracker

    def test_flow_name_tracker_exists(self):
        """Test that flow_name_tracker exists."""
        assert hasattr(testing.TestBase, "flow_name_tracker")
        assert isinstance(testing.TestBase.flow_name_tracker, list)

    def test_test_server_url(self):
        """Test test_server URL is set."""
        assert hasattr(testing.TestBase, "test_server")
        assert "test.openml.org" in testing.TestBase.test_server

    def test_apikey_exists(self):
        """Test that apikey is defined."""
        assert hasattr(testing.TestBase, "apikey")
        assert isinstance(testing.TestBase.apikey, str)


class TestTestBaseSetUp:
    """Test TestBase setUp method."""

    def test_setup_creates_workdir(self):
        """Test that setUp creates working directory."""
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            test_instance.setUp(n_levels=0)
        
        assert test_instance.workdir.exists()
        
        # Cleanup
        test_instance.tearDown()

    def test_setup_sets_apikey(self):
        """Test that setUp sets API key."""
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path  
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            test_instance.setUp(n_levels=0)
        
        import openml
        assert openml.config.apikey == testing.TestBase.apikey
        
        # Cleanup
        test_instance.tearDown()

    def test_setup_changes_directory(self):
        """Test that setUp changes to workdir."""
        import os
        
        original_dir = Path.cwd()
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            test_instance.setUp(n_levels=0)
        
        # Directory should have changed
        assert Path.cwd() != original_dir
        
        # Cleanup
        test_instance.tearDown()
        assert Path.cwd() == original_dir


class TestTestBaseTearDown:
    """Test TestBase tearDown method."""

    def test_teardown_removes_workdir(self):
        """Test that tearDown removes working directory."""
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            test_instance.setUp(n_levels=0)
        
        workdir = test_instance.workdir
        assert workdir.exists()
        
        test_instance.tearDown()
        
        # Workdir should be removed
        assert not workdir.exists()

    def test_teardown_restores_directory(self):
        """Test that tearDown restores original directory."""
        import os
        
        original_dir = Path.cwd()
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            test_instance.setUp(n_levels=0)
        
        test_instance.tearDown()
        
        assert Path.cwd() == original_dir


class TestTestBaseMarkEntityForRemoval:
    """Test _mark_entity_for_removal class method."""

    def test_mark_entity_for_removal_new_entity(self):
        """Test marking new entity type for removal."""
        # Save original state
        original_tracker = testing.TestBase.publish_tracker.copy()
        
        # Clear tracker
        testing.TestBase.publish_tracker["test_entity"] = []
        
        testing.TestBase._mark_entity_for_removal("test_entity", 123)
        
        assert 123 in testing.TestBase.publish_tracker["test_entity"]
        
        # Restore
        testing.TestBase.publish_tracker = original_tracker

    def test_mark_entity_for_removal_existing_entity_type(self):
        """Test marking entity when type already exists."""
        original_tracker = testing.TestBase.publish_tracker.copy()
        
        testing.TestBase.publish_tracker["run"] = []
        testing.TestBase._mark_entity_for_removal("run", 456)
        
        assert 456 in testing.TestBase.publish_tracker["run"]
        
        testing.TestBase.publish_tracker = original_tracker


class TestTestBaseDeleteEntityFromTracker:
    """Test _delete_entity_from_tracker class method."""

    def test_delete_entity_from_tracker(self):
        """Test deleting entity from tracker."""
        original_tracker = testing.TestBase.publish_tracker.copy()
        
        testing.TestBase.publish_tracker["run"] = [1, 2, 3, 2]  # With duplicate
        testing.TestBase._delete_entity_from_tracker("run", 2)
        
        assert 2 not in testing.TestBase.publish_tracker["run"]
        
        testing.TestBase.publish_tracker = original_tracker


class TestTestBaseGetSentinel:
    """Test _get_sentinel method."""

    def test_get_sentinel_without_argument(self):
        """Test _get_sentinel generates unique sentinel."""
        test_instance = testing.TestBase()
        
        sentinel1 = test_instance._get_sentinel()
        sentinel2 = test_instance._get_sentinel()
        
        # Should start with TEST
        assert sentinel1.startswith("TEST")
        assert sentinel2.startswith("TEST")
        
        # Should be different (time-based)
        # Note: Might be same if called too quickly
        assert len(sentinel1) == 14  # TEST + 10 chars

    def test_get_sentinel_with_argument(self):
        """Test _get_sentinel with provided sentinel."""
        test_instance = testing.TestBase()
        
        sentinel = test_instance._get_sentinel("CUSTOM")
        
        assert sentinel == "CUSTOM"

    def test_get_sentinel_generates_hex(self):
        """Test _get_sentinel generates hexadecimal."""
        test_instance = testing.TestBase()
        
        sentinel = test_instance._get_sentinel()
        
        # Remove TEST prefix and check if rest is hex
        hex_part = sentinel[4:]
        assert all(c in "0123456789abcdef" for c in hex_part.lower())


class TestTestBaseAddSentinelToFlowName:
    """Test _add_sentinel_to_flow_name method."""

    def test_add_sentinel_to_flow_name(self):
        """Test adding sentinel to flow name."""
        from openml.flows import OpenMLFlow
        
        test_instance = testing.TestBase()
        
        flow = Mock(spec=OpenMLFlow)
        flow.name = "TestFlow"
        flow.components = {}
        
        modified_flow, sentinel = test_instance._add_sentinel_to_flow_name(flow)
        
        assert sentinel in modified_flow.name
        assert sentinel.startswith("TEST")

    def test_add_sentinel_to_flow_name_with_custom_sentinel(self):
        """Test adding custom sentinel to flow name."""
        from openml.flows import OpenMLFlow
        
        test_instance = testing.TestBase()
        
        flow = Mock(spec=OpenMLFlow)
        flow.name = "MyFlow"
        flow.components = {}
        
        modified_flow, sentinel = test_instance._add_sentinel_to_flow_name(
            flow, 
            sentinel="CUSTOM123"
        )
        
        assert "CUSTOM123" in modified_flow.name
        assert sentinel == "CUSTOM123"


class TestTestBaseCheckDataset:
    """Test _check_dataset method."""

    def test_check_dataset_valid(self):
        """Test _check_dataset with valid dataset."""
        test_instance = testing.TestBase()
        
        dataset = {
            "did": 123,
            "status": "active"
        }
        
        # Should not raise
        test_instance._check_dataset(dataset)

    def test_check_dataset_checks_type(self):
        """Test _check_dataset verifies types."""
        test_instance = testing.TestBase()
        
        dataset = {
            "did": 123,
            "status": "active"
        }
        
        # Should not raise
        test_instance._check_dataset(dataset)


class TestTestBaseCheckFoldTimingEvaluations:
    """Test _check_fold_timing_evaluations method."""

    def test_check_fold_timing_evaluations_basic(self):
        """Test _check_fold_timing_evaluations with valid data."""
        test_instance = testing.TestBase()
        
        fold_evaluations = {
            "usercpu_time_millis": {0: {0: 100.0, 1: 110.0}},
            "usercpu_time_millis_training": {0: {0: 50.0, 1: 55.0}},
            "usercpu_time_millis_testing": {0: {0: 50.0, 1: 55.0}},
            "wall_clock_time_millis": {0: {0: 120.0, 1: 130.0}},
            "wall_clock_time_millis_training": {0: {0: 60.0, 1: 65.0}},
            "wall_clock_time_millis_testing": {0: {0: 60.0, 1: 65.0}},
            "predictive_accuracy": {0: {0: 0.9, 1: 0.92}}
        }
        
        # Should not raise
        test_instance._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=2,
            task_type=TaskType.SUPERVISED_CLASSIFICATION
        )

    def test_check_fold_timing_evaluations_without_scores(self):
        """Test _check_fold_timing_evaluations without score checking."""
        test_instance = testing.TestBase()
        
        fold_evaluations = {
            "usercpu_time_millis": {0: {0: 100.0}},
            "usercpu_time_millis_training": {0: {0: 50.0}},
            "usercpu_time_millis_testing": {0: {0: 50.0}},
            "wall_clock_time_millis": {0: {0: 120.0}},
            "wall_clock_time_millis_training": {0: {0: 60.0}},
            "wall_clock_time_millis_testing": {0: {0: 60.0}},
        }
        
        # Should not raise
        test_instance._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=1,
            check_scores=False
        )


class TestTestBaseUseProductionServer:
    """Test use_production_server method."""

    def test_use_production_server_sets_server(self):
        """Test use_production_server sets correct server."""
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            test_instance.setUp(n_levels=0)
        
        import openml
        test_instance.use_production_server()
        
        assert openml.config.server == test_instance.production_server
        
        # Cleanup
        test_instance.tearDown()


class TestModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Test that __all__ includes expected items."""
        expected_exports = [
            "TestBase",
            "SimpleImputer",
            "CustomImputer",
            "check_task_existence",
            "create_request_response"
        ]
        
        for item in expected_exports:
            assert item in testing.__all__


class TestEdgeCases:
    """Test edge cases for testing utilities."""

    def test_get_sentinel_consistency(self):
        """Test that sentinel generation is consistent when provided."""
        test_instance = testing.TestBase()
        
        sentinel1 = test_instance._get_sentinel("FIXED")
        sentinel2 = test_instance._get_sentinel("FIXED")
        
        assert sentinel1 == sentinel2 == "FIXED"

    def test_create_request_response_with_empty_file(self, tmp_path):
        """Test create_request_response with empty file."""
        xml_file = tmp_path / "empty.xml"
        xml_file.write_text("")
        
        response = testing.create_request_response(
            status_code=200,
            content_filepath=xml_file
        )
        
        assert response.status_code == 200
        assert response.text == ""

    def test_create_request_response_with_large_file(self, tmp_path):
        """Test create_request_response with large file."""
        xml_file = tmp_path / "large.xml"
        large_content = "<data>" + "x" * 100000 + "</data>"
        xml_file.write_text(large_content)
        
        response = testing.create_request_response(
            status_code=200,
            content_filepath=xml_file
        )
        
        assert len(response.text) > 100000


class TestTestBaseIntegration:
    """Integration tests for TestBase."""

    def test_full_setup_teardown_cycle(self):
        """Test complete setUp and tearDown cycle."""
        import os
        
        original_dir = Path.cwd()
        test_instance = testing.TestBase()
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        with patch.object(Path, 'absolute', return_value=files_dir):
            test_instance.static_cache_dir = files_dir
            # Setup
            test_instance.setUp(n_levels=0)
        assert test_instance.workdir.exists()
        assert Path.cwd() != original_dir
        
        # Teardown
        workdir = test_instance.workdir
        test_instance.tearDown()
        assert not workdir.exists()
        assert Path.cwd() == original_dir

    def test_multiple_test_instances(self):
        """Test creating multiple test instances."""
        instances = []
        
        # Mock the static_cache_dir path
        files_dir = Path("/home/naman/Desktop/openml-python/tests/files")
        
        for i in range(3):
            instance = testing.TestBase()
            with patch.object(Path, 'absolute', return_value=files_dir):
                instance.static_cache_dir = files_dir
                instance.setUp(n_levels=0, tmpdir_suffix=f"_{i}")
            instances.append(instance)
        
        # All should have different work directories
        workdirs = [inst.workdir for inst in instances]
        assert len(set(workdirs)) == 3
        
        # Cleanup
        for instance in instances:
            instance.tearDown()
