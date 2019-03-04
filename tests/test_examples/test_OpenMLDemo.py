import os
import shutil
import sys

import matplotlib
matplotlib.use('AGG')
import nbformat
from nbconvert.exporters import export
from nbconvert.exporters.python import PythonExporter

import unittest.mock as mock

from unittest import skip
import openml._api_calls
import openml.config
from openml.testing import TestBase

_perform_api_call = openml._api_calls._perform_api_call


class OpenMLDemoTest(TestBase):
    def setUp(self):
        super(OpenMLDemoTest, self).setUp()

        python_version = sys.version_info[0]
        self.kernel_name = 'python%d' % python_version
        self.this_file_directory = os.path.dirname(__file__)
        self.notebook_output_directory = os.path.join(
            self.this_file_directory, '.out')

        try:
            shutil.rmtree(self.notebook_output_directory)
        except OSError:
            pass

        try:
            os.makedirs(self.notebook_output_directory)
        except OSError:
            pass

    def _tst_notebook(self, notebook_name):

        notebook_filename = os.path.abspath(os.path.join(
            self.this_file_directory, '..', '..', 'examples', notebook_name))

        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)

        python_nb, metadata = export(PythonExporter, nb)

        # Remove magic lines manually
        python_nb = '\n'.join([
            line for line in python_nb.split('\n')
            if 'get_ipython().run_line_magic(' not in line
        ])

        exec(python_nb)

    @skip
    @mock.patch('openml._api_calls._perform_api_call')
    def test_tutorial_openml(self, patch):
        def side_effect(*args, **kwargs):
            if (
                args[0].endswith('/run/')
                and kwargs['file_elements'] is not None
            ):
                return """<oml:upload_run>
    <oml:run_id>1</oml:run_id>
</oml:upload_run>
                """
            else:
                return _perform_api_call(*args, **kwargs)
        patch.side_effect = side_effect

        openml.config.server = self.production_server
        self._tst_notebook('OpenML_Tutorial.ipynb')
        self.assertGreater(patch.call_count, 100)

    @skip("Deleted tutorial file")
    def test_tutorial_dataset(self):

        self._tst_notebook('Dataset_import.ipynb')
