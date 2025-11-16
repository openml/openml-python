# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.study.study module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openml.study.study import BaseStudy, OpenMLBenchmarkSuite, OpenMLStudy


class TestBaseStudyInit:
    """Test BaseStudy initialization."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        study = BaseStudy(
            study_id=100,
            alias="test-study",
            main_entity_type="run",
            benchmark_suite=200,
            name="Test Study",
            description="A test study",
            status="active",
            creation_date="2023-01-01T12:00:00",
            creator=300,
            tags=[{"tag": "test", "window_start": 0, "write_access": "public"}],
            data=[1, 2, 3],
            tasks=[10, 20, 30],
            flows=[100, 200],
            runs=[1000, 2000, 3000],
            setups=[500, 600]
        )
        
        assert study.study_id == 100
        assert study.alias == "test-study"
        assert study.main_entity_type == "run"
        assert study.benchmark_suite == 200
        assert study.name == "Test Study"
        assert study.description == "A test study"
        assert study.status == "active"
        assert study.creation_date == "2023-01-01T12:00:00"
        assert study.creator == 300
        assert study.tags is not None
        assert study.data == [1, 2, 3]
        assert study.tasks == [10, 20, 30]
        assert study.flows == [100, 200]
        assert study.runs == [1000, 2000, 3000]
        assert study.setups == [500, 600]

    def test_init_with_none_values(self):
        """Test initialization with None values."""
        study = BaseStudy(
            study_id=None,
            alias=None,
            main_entity_type="task",
            benchmark_suite=None,
            name="Minimal Study",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        assert study.study_id is None
        assert study.alias is None
        assert study.benchmark_suite is None


class TestBaseStudyProperties:
    """Test BaseStudy properties."""

    def test_id_property(self):
        """Test id property returns study_id."""
        study = BaseStudy(
            study_id=123,
            alias=None,
            main_entity_type="run",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        assert study.id == 123

    def test_id_property_when_none(self):
        """Test id property when study_id is None."""
        study = BaseStudy(
            study_id=None,
            alias=None,
            main_entity_type="run",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        assert study.id is None

    def test_entity_letter(self):
        """Test _entity_letter returns 's'."""
        assert BaseStudy._entity_letter() == "s"


class TestBaseStudyGetReprBodyFields:
    """Test _get_repr_body_fields method."""

    def test_repr_body_fields_complete(self):
        """Test _get_repr_body_fields with all fields populated."""
        study = BaseStudy(
            study_id=100,
            alias="study-alias",
            main_entity_type="run",
            benchmark_suite=None,
            name="Complete Study",
            description="Full description",
            status="active",
            creation_date="2023-06-15T10:30:00",
            creator=500,
            tags=None,
            data=[1, 2, 3],
            tasks=[10, 20],
            flows=[100],
            runs=[1000, 2000],
            setups=None
        )
        
        with patch("openml.config.get_server_base_url", return_value="https://openml.org"):
            fields = study._get_repr_body_fields()
            
            field_dict = dict(fields)
            assert "ID" in field_dict
            assert field_dict["ID"] == 100
            assert "Name" in field_dict
            assert field_dict["Name"] == "Complete Study"
            assert "Status" in field_dict
            assert field_dict["Status"] == "active"
            assert "# of Data" in field_dict
            assert field_dict["# of Data"] == 3

    def test_repr_body_fields_minimal(self):
        """Test _get_repr_body_fields with minimal fields."""
        study = BaseStudy(
            study_id=None,
            alias=None,
            main_entity_type="task",
            benchmark_suite=None,
            name="Minimal",
            description="Desc",
            status="in_preparation",
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        fields = study._get_repr_body_fields()
        field_dict = dict(fields)
        
        assert "Name" in field_dict
        assert "Status" in field_dict
        assert "Main Entity Type" in field_dict
        # Should not have ID when study_id is None
        assert "ID" not in field_dict


class TestBaseStudyToDict:
    """Test _to_dict method."""

    def test_to_dict_with_tasks(self):
        """Test _to_dict with tasks."""
        study = BaseStudy(
            study_id=100,
            alias="test",
            main_entity_type="task",
            benchmark_suite=None,
            name="Task Study",
            description="Description",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=[1, 2, 3],
            flows=None,
            runs=None,
            setups=None
        )
        
        result = study._to_dict()
        
        assert "oml:study" in result
        assert "oml:name" in result["oml:study"]
        assert result["oml:study"]["oml:name"] == "Task Study"
        assert "oml:tasks" in result["oml:study"]
        assert result["oml:study"]["oml:tasks"]["oml:task_id"] == [1, 2, 3]

    def test_to_dict_with_runs(self):
        """Test _to_dict with runs."""
        study = BaseStudy(
            study_id=200,
            alias="run-study",
            main_entity_type="run",
            benchmark_suite=None,
            name="Run Study",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=[100, 200, 300],
            setups=None
        )
        
        result = study._to_dict()
        
        assert "oml:runs" in result["oml:study"]
        assert result["oml:study"]["oml:runs"]["oml:run_id"] == [100, 200, 300]

    def test_to_dict_minimal(self):
        """Test _to_dict with minimal data."""
        study = BaseStudy(
            study_id=None,
            alias=None,
            main_entity_type="run",
            benchmark_suite=None,
            name="Minimal",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        result = study._to_dict()
        
        assert "oml:study" in result
        assert "oml:name" in result["oml:study"]
        assert "@xmlns:oml" in result["oml:study"]


class TestBaseStudyParsePublishResponse:
    """Test _parse_publish_response method."""

    def test_parse_publish_response(self):
        """Test parsing publish response."""
        study = BaseStudy(
            study_id=None,
            alias=None,
            main_entity_type="run",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        xml_response = {
            "oml:study_upload": {
                "oml:id": "999"
            }
        }
        
        study._parse_publish_response(xml_response)
        
        assert study.study_id == 999
        assert study.id == 999


class TestBaseStudyTags:
    """Test tag-related methods."""

    def test_push_tag_raises_not_implemented(self):
        """Test that push_tag raises NotImplementedError."""
        study = BaseStudy(
            study_id=1,
            alias=None,
            main_entity_type="run",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        with pytest.raises(NotImplementedError, match="not \\(yet\\) supported"):
            study.push_tag("test_tag")

    def test_remove_tag_raises_not_implemented(self):
        """Test that remove_tag raises NotImplementedError."""
        study = BaseStudy(
            study_id=1,
            alias=None,
            main_entity_type="run",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        with pytest.raises(NotImplementedError, match="not \\(yet\\) supported"):
            study.remove_tag("test_tag")


class TestOpenMLStudyInit:
    """Test OpenMLStudy initialization."""

    def test_init_sets_main_entity_type_to_run(self):
        """Test that OpenMLStudy sets main_entity_type to 'run'."""
        study = OpenMLStudy(
            study_id=100,
            alias="test",
            benchmark_suite=None,
            name="Study",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=200,
            tags=None,
            data=[1],
            tasks=[2],
            flows=[3],
            runs=[4],
            setups=[5]
        )
        
        assert study.main_entity_type == "run"

    def test_init_with_benchmark_suite(self):
        """Test initialization with benchmark suite."""
        study = OpenMLStudy(
            study_id=100,
            alias="test",
            benchmark_suite=500,
            name="Study",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=200,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=[1, 2, 3],
            setups=None
        )
        
        assert study.benchmark_suite == 500


class TestOpenMLBenchmarkSuiteInit:
    """Test OpenMLBenchmarkSuite initialization."""

    def test_init_sets_main_entity_type_to_task(self):
        """Test that OpenMLBenchmarkSuite sets main_entity_type to 'task'."""
        suite = OpenMLBenchmarkSuite(
            suite_id=100,
            alias="test-suite",
            name="Benchmark",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=200,
            tags=None,
            data=[1, 2],
            tasks=[10, 20, 30]
        )
        
        assert suite.main_entity_type == "task"

    def test_init_sets_benchmark_suite_to_none(self):
        """Test that OpenMLBenchmarkSuite sets benchmark_suite to None."""
        suite = OpenMLBenchmarkSuite(
            suite_id=100,
            alias="test",
            name="Suite",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=200,
            tags=None,
            data=None,
            tasks=[1, 2]
        )
        
        assert suite.benchmark_suite is None

    def test_init_sets_flows_and_runs_to_none(self):
        """Test that OpenMLBenchmarkSuite sets flows, runs, setups to None."""
        suite = OpenMLBenchmarkSuite(
            suite_id=100,
            alias="test",
            name="Suite",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=200,
            tags=None,
            data=None,
            tasks=[1]
        )
        
        assert suite.flows is None
        assert suite.runs is None
        assert suite.setups is None


class TestStudyEdgeCases:
    """Test edge cases for study classes."""

    def test_empty_lists(self):
        """Test with empty lists for collections."""
        study = OpenMLStudy(
            study_id=1,
            alias="test",
            benchmark_suite=None,
            name="Empty",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=1,
            tags=[],
            data=[],
            tasks=[],
            flows=[],
            runs=[],
            setups=[]
        )
        
        assert study.data == []
        assert study.tasks == []
        assert study.flows == []
        assert study.runs == []
        assert study.setups == []

    def test_large_lists(self):
        """Test with large lists of IDs."""
        large_list = list(range(10000))
        study = OpenMLStudy(
            study_id=1,
            alias="large",
            benchmark_suite=None,
            name="Large Study",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=1,
            tags=None,
            data=large_list,
            tasks=large_list,
            flows=large_list,
            runs=large_list,
            setups=large_list
        )
        
        assert len(study.data) == 10000
        assert len(study.runs) == 10000

    def test_unicode_in_name_and_description(self):
        """Test with unicode characters."""
        study = OpenMLStudy(
            study_id=1,
            alias="unicode-study",
            benchmark_suite=None,
            name="研究名称",
            description="説明文 with 日本語",
            status="active",
            creation_date="2023-01-01",
            creator=1,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        assert "研究" in study.name
        assert "日本語" in study.description

    def test_special_characters_in_alias(self):
        """Test with special characters in alias."""
        study = OpenMLStudy(
            study_id=1,
            alias="test-study_v1.0",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=1,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            runs=None,
            setups=None
        )
        
        assert study.alias == "test-study_v1.0"

    def test_different_status_values(self):
        """Test with different status values."""
        statuses = ["in_preparation", "active", "deactivated"]
        
        for status in statuses:
            study = OpenMLStudy(
                study_id=1,
                alias="test",
                benchmark_suite=None,
                name="Test",
                description="Desc",
                status=status,
                creation_date="2023-01-01",
                creator=1,
                tags=None,
                data=None,
                tasks=None,
                flows=None,
                runs=None,
                setups=None
            )
            
            assert study.status == status


class TestStudyIntegration:
    """Integration tests for study classes."""

    def test_study_repr(self):
        """Test that study __repr__ works."""
        study = OpenMLStudy(
            study_id=100,
            alias="test",
            benchmark_suite=None,
            name="Integration Test",
            description="Description",
            status="active",
            creation_date="2023-01-01T12:00:00",
            creator=200,
            tags=None,
            data=[1, 2, 3],
            tasks=[10, 20],
            flows=[100],
            runs=[1000, 2000, 3000],
            setups=None
        )
        
        repr_str = repr(study)
        
        assert "OpenML" in repr_str
        assert "Study" in repr_str
        assert "Integration Test" in repr_str

    def test_benchmark_suite_repr(self):
        """Test that benchmark suite __repr__ works."""
        suite = OpenMLBenchmarkSuite(
            suite_id=50,
            alias="suite",
            name="Test Suite",
            description="Desc",
            status="active",
            creation_date="2023-01-01",
            creator=100,
            tags=None,
            data=[1, 2],
            tasks=[10, 20, 30, 40, 50]
        )
        
        repr_str = repr(suite)
        
        assert "OpenML" in repr_str
        assert "Benchmark Suite" in repr_str or "Suite" in repr_str
        assert "Test Suite" in repr_str

    def test_study_to_dict_and_back(self):
        """Test converting study to dict."""
        original = OpenMLStudy(
            study_id=None,
            alias="test",
            benchmark_suite=None,
            name="Test",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=[1, 2, 3],
            flows=None,
            runs=[100, 200],
            setups=None
        )
        
        dict_repr = original._to_dict()
        
        assert "oml:study" in dict_repr
        assert dict_repr["oml:study"]["oml:name"] == "Test"

    def test_benchmark_suite_to_dict(self):
        """Test converting benchmark suite to dict."""
        suite = OpenMLBenchmarkSuite(
            suite_id=None,
            alias="suite",
            name="Suite",
            description="Desc",
            status=None,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=[1, 2, 3, 4, 5]
        )
        
        dict_repr = suite._to_dict()
        
        assert "oml:study" in dict_repr
        assert "oml:tasks" in dict_repr["oml:study"]

    def test_multiple_studies(self):
        """Test creating multiple study objects."""
        studies = []
        
        for i in range(10):
            study = OpenMLStudy(
                study_id=i,
                alias=f"study-{i}",
                benchmark_suite=None,
                name=f"Study {i}",
                description=f"Description {i}",
                status="active",
                creation_date="2023-01-01",
                creator=i * 10,
                tags=None,
                data=None,
                tasks=None,
                flows=None,
                runs=[i * 100],
                setups=None
            )
            studies.append(study)
        
        assert len(studies) == 10
        assert all(isinstance(s, OpenMLStudy) for s in studies)
        assert all(s.main_entity_type == "run" for s in studies)
