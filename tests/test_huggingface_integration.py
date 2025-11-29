import unittest
from unittest.mock import MagicMock, patch
import pytest
import sys

# Mock modules if they don't exist, so we can test the logic
# This needs to be done before importing the integration module if we want to force-enable it
# But the integration module does a try-import.

from openml.runs import OpenMLRun
import openml.extensions.huggingface_integration as hf_int

class TestHuggingFaceIntegration(unittest.TestCase):

    def setUp(self):
        self.run = OpenMLRun(task_id=1, flow_id=1, dataset_id=1)
        self.run.run_id = 123
        self.run.tags = []

    def test_is_hf_transformer_no_deps(self):
        # Force _HF_AVAILABLE to False
        with patch("openml.extensions.huggingface_integration._HF_AVAILABLE", False):
            self.assertFalse(hf_int.is_hf_transformer(MagicMock()))

    def test_push_model_no_deps(self):
        with patch("openml.extensions.huggingface_integration._HF_AVAILABLE", False):
            model = MagicMock()
            run = hf_int.push_model_to_hub_for_run(model, self.run, "repo")
            self.assertEqual(run.tags, [])

    def test_load_model_no_deps(self):
        with patch("openml.extensions.huggingface_integration._HF_AVAILABLE", False):
            with self.assertRaises(ImportError):
                hf_int.load_model_from_run(123)

    @patch("openml.extensions.huggingface_integration._HF_AVAILABLE", True)
    def test_is_hf_transformer_with_deps(self):
        # We need to mock PreTrainedModel in the module
        MockPTM = MagicMock()
        with patch("openml.extensions.huggingface_integration.PreTrainedModel", MockPTM):
            model = MockPTM()
            # isinstance check in the module needs to work. 
            # Since we patched the name 'PreTrainedModel' in the module, 
            # if we create an instance of that mock, isinstance might not work as expected 
            # if the module uses the *real* class it imported (or failed to import).
            
            # If the module successfully imported PreTrainedModel, it holds a reference to the real class.
            # If it failed, it holds 'object'.
            
            # If we want to test the True path, we should rely on the module's reference.
            pass 
            # This is getting complicated to test "with deps" if they aren't actually there.
            # I will rely on the fact that if they are there, we test it.
            # If not, we skip the "with deps" tests.

@pytest.mark.skipif(not hf_int._HF_AVAILABLE, reason="Hugging Face dependencies not installed")
class TestHuggingFaceIntegrationWithDeps(unittest.TestCase):
    
    def setUp(self):
        self.run = OpenMLRun(task_id=1, flow_id=1, dataset_id=1)
        self.run.run_id = 123
        self.run.tags = []

    def test_is_hf_transformer(self):
        from transformers import PreTrainedModel
        # Create a dummy subclass
        class DummyModel(PreTrainedModel):
            def __init__(self):
                # Minimal init to satisfy PreTrainedModel if needed, 
                # but usually we can just mock or pass dummy config
                self.config = MagicMock()
                
        model = DummyModel()
        self.assertTrue(hf_int.is_hf_transformer(model))
        self.assertFalse(hf_int.is_hf_transformer("string"))

    @patch("openml.extensions.huggingface_integration.HfApi")
    def test_push_model_to_hub_for_run(self, MockHfApi):
        from transformers import PreTrainedModel
        
        model = MagicMock(spec=PreTrainedModel)
        
        # Mock HfApi
        mock_api = MockHfApi.return_value
        mock_commit = MagicMock()
        mock_commit.commit_id = "sha123"
        mock_api.list_repo_commits.return_value = [mock_commit]
        
        run = hf_int.push_model_to_hub_for_run(model, self.run, "user/repo")
        
        model.push_to_hub.assert_called_with("user/repo", commit_message="OpenML Run 123", token=None)
        self.assertIn("hf_uri=hf://user/repo@sha123", run.tags)
        self.assertIn("hf-integrated", run.tags)

    @patch("openml.extensions.huggingface_integration.AutoModel")
    @patch("openml.runs.get_run")
    def test_load_model_from_run(self, mock_get_run, MockAutoModel):
        self.run.tags = ["hf_uri=hf://user/repo@sha123"]
        mock_get_run.return_value = self.run
        
        hf_int.load_model_from_run(123)
        
        MockAutoModel.from_pretrained.assert_called_with("user/repo", revision="sha123", token=None)

    @patch("openml.runs.get_run")
    def test_load_model_from_run_missing_tag(self, mock_get_run):
        mock_get_run.return_value = self.run
        with self.assertRaises(ValueError):
            hf_int.load_model_from_run(123)

    @patch("openml.runs.get_run")
    def test_load_model_from_run_bad_uri(self, mock_get_run):
        self.run.tags = ["hf_uri=hf://bad_uri"]
        mock_get_run.return_value = self.run
        with self.assertRaises(ValueError):
            hf_int.load_model_from_run(123)
