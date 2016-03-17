import os
import shutil
import unittest
import openml


class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    A config file with the username and password must be present to test the
    API calls.
    """

    def setUp(self):
        self.cwd = os.getcwd()
        workdir = os.path.dirname(os.path.abspath(__file__))
        self.workdir = os.path.join(workdir, "tmp")
        try:
            shutil.rmtree(self.workdir)
        except:
            pass

        os.mkdir(self.workdir)
        os.chdir(self.workdir)

        self.cached = True
        # amueller's read/write key that he will throw away later
        apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.set_apikey(apikey)
        openml.config.set_cachedir(self.workdir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir)

__all__ = ['TestBase']
