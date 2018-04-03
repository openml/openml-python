import os
import shutil
import sys

from IPython import get_ipython
import nbformat
from nbconvert.exporters import export
from nbconvert.exporters.python import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

import openml.config
from openml.testing import TestBase


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
        except:
            pass

        try:
            os.makedirs(self.notebook_output_directory)
        except:
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
        print(type(python_nb), python_nb)

        exec(python_nb)

    def test_tutorial(self):
        openml.config.server = self.production_server
        self._tst_notebook('OpenML_Tutorial.ipynb')
