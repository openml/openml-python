import os
import unittest
import shutil
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


class OpenMLDemoTest(unittest.TestCase):
    def setUp(self):
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

    @unittest.skip('Local skipy to save time...')
    def test_notebook(self):
        notebook_name = 'OpenMLDemo.ipynb'

        notebook_filename = os.path.abspath(os.path.join(
            self.this_file_directory, '..', '..', 'examples', notebook_name))
        notebook_filename_out = os.path.join(
            self.notebook_output_directory, notebook_name)

        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
            nb.metadata.get('kernelspec', {})['name'] = self.kernel_name
            ep = ExecutePreprocessor(kernel_name=self.kernel_name)

            try:
                ep.preprocess(nb, {'metadata': {'path': self.this_file_directory}})
            except CellExecutionError as e:
                msg = 'Error executing the notebook "%s". ' % notebook_filename
                msg += 'See notebook "%s" for the traceback.\n\n' % notebook_filename_out
                msg += e.traceback
                self.fail(msg)
            finally:
                with open(notebook_filename_out, mode='wt') as f:
                    nbformat.write(nb, f)
