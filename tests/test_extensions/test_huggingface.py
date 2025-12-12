"""Tests for HuggingFace integration."""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openml.exceptions import PyOpenMLError
from openml.flows import OpenMLFlow


class TestHuggingFaceIntegration:
    """Test suite for HuggingFace Hub integration."""

    @pytest.fixture
    def mock_flow(self):
        """Create a mock OpenML flow for testing."""
        flow = MagicMock(spec=OpenMLFlow)
        flow.flow_id = 12345
        flow.name = "sklearn.ensemble.RandomForestClassifier"
        flow.description = "A random forest classifier"
        flow.external_version = "1.0.0"
        flow.dependencies = "scikit-learn==1.3.0"
        flow.parameters = {"n_estimators": "100", "max_depth": "10"}
        flow.model = MagicMock()  # Mock model object
        return flow

    def test_upload_flow_without_flow_id_raises_error(self, mock_flow):
        """Test that uploading a flow without flow_id raises PyOpenMLError."""
        from openml.extensions.huggingface import upload_flow_to_huggingface

        mock_flow.flow_id = None

        with pytest.raises(PyOpenMLError, match="must be published"):
            upload_flow_to_huggingface(
                flow=mock_flow,
                repo_id="test/repo",
                token="fake_token",
            )

    def test_upload_flow_without_model_raises_error(self, mock_flow):
        """Test that uploading a flow without model raises PyOpenMLError."""
        from openml.extensions.huggingface import upload_flow_to_huggingface

        mock_flow.model = None

        with pytest.raises(PyOpenMLError, match="must have a model"):
            upload_flow_to_huggingface(
                flow=mock_flow,
                repo_id="test/repo",
                token="fake_token",
            )

    @patch("openml.extensions.huggingface.functions.pickle.dump")
    @patch("openml.extensions.huggingface.functions.create_repo")
    @patch("openml.extensions.huggingface.functions.HfApi")
    def test_upload_creates_correct_files(
        self, mock_hf_api, mock_create_repo, mock_pickle_dump, mock_flow
    ):
        """Test that upload creates model.pkl, metadata.json, and README.md."""
        from openml.extensions.huggingface import upload_flow_to_huggingface

        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        upload_flow_to_huggingface(
            flow=mock_flow,
            repo_id="test/repo",
            token="fake_token",
        )

        # Verify create_repo was called
        mock_create_repo.assert_called_once()

        # Verify pickle.dump was called (for the model)
        mock_pickle_dump.assert_called_once()

        # Verify upload_file was called 3 times (model, metadata, README)
        assert mock_api_instance.upload_file.call_count == 3

        # Verify the files have correct names
        call_args_list = mock_api_instance.upload_file.call_args_list
        uploaded_files = [call.kwargs.get("path_in_repo") for call in call_args_list]

        assert "model.pkl" in uploaded_files
        assert "openml_metadata.json" in uploaded_files
        assert "README.md" in uploaded_files

    def test_model_card_generation(self, mock_flow):
        """Test that model card is generated correctly."""
        from openml.extensions.huggingface.functions import _create_model_card

        card = _create_model_card(mock_flow)

        assert "sklearn.ensemble.RandomForestClassifier" in card
        assert "12345" in card
        assert "n_estimators" in card
        assert "https://www.openml.org/f/12345" in card

    @patch("openml.extensions.huggingface.functions.hf_hub_download")
    def test_download_flow_loads_correct_files(self, mock_download):
        """Test that download correctly loads model and metadata."""
        from openml.extensions.huggingface import download_flow_from_huggingface

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create mock model file
            model_path = tmpdir_path / "model.pkl"
            mock_model = {"type": "RandomForest"}
            with model_path.open("wb") as f:
                pickle.dump(mock_model, f)

            # Create mock metadata file
            metadata_path = tmpdir_path / "openml_metadata.json"
            mock_metadata = {"openml_flow_id": 12345}
            with metadata_path.open("w") as f:
                json.dump(mock_metadata, f)

            # Mock hf_hub_download to return our temp files
            def side_effect(repo_id, filename, **kwargs):
                if filename == "model.pkl":
                    return str(model_path)
                elif filename == "openml_metadata.json":
                    return str(metadata_path)
                return None

            mock_download.side_effect = side_effect

            # Test download
            result = download_flow_from_huggingface("test/repo")

            assert result["model"] == mock_model
            assert result["metadata"]["openml_flow_id"] == 12345

    def test_config_initialization(self):
        """Test that config initializes correctly."""
        from openml.extensions.huggingface.config import get_config, reset_config

        reset_config()
        config = get_config()

        assert config.model_filename == "model.pkl"
        assert config.metadata_filename == "openml_metadata.json"
        assert config.readme_filename == "README.md"
        assert config.cache_dir is not None

    def test_config_cache_directory_setting(self):
        """Test setting custom cache directory."""
        from openml.extensions.huggingface.config import (
            get_config,
            set_cache_directory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            set_cache_directory(tmpdir)
            config = get_config()

            assert str(config.cache_dir) == tmpdir
            assert config.cache_dir.exists()