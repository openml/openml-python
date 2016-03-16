from .apiconnector import APIConnector
import os
import shutil
import unittest


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
        # amueller's read only api key
        apikey = "4e36ac93097f979f921b74b02a600f34"

        self.connector = APIConnector(cache_directory=self.workdir,
                                      apikey=apikey)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir)

__all__ = ['TestBase']
