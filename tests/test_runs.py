import os
import tempfile

import pytest
import openml
from openml.runs.run import OpenMLRun

def test_publish_run_with_additional_files(monkeypatch):
    """Ensure publish_run uploads each additional file."""
    # capture calls to upload_run_file
    calls = []
    def fake_upload_run_file(run_id, file_path):
        calls.append((run_id, file_path))

    # Monkey-patch the client’s api_calls.upload_run_file
    # OpenMLRun._client is a singleton, so patch its _api_calls directly
    monkeypatch.setattr(
        openml.runs.run.OpenMLRun._client._api_calls,
        "upload_run_file",
        staticmethod(fake_upload_run_file),
    )

    # Create a dummy run (you can pick any valid task_id/flow_id/seed)
    run = OpenMLRun(task_id=1, flow_id=1, seed=1)

    # Write a temp file to pass as additional_files
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.write(b"dummy")
    tf.flush()
    tf.close()

    # Publish the run with our extra file
    rid = run.publish_run(additional_files=[tf.name])

    # We should have exactly one upload call with our run id and file path
    assert len(calls) == 1
    assert calls[0][0] == rid
    assert calls[0][1] == tf.name

    # Clean up
    os.remove(tf.name)

