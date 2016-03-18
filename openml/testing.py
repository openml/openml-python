import os
import shutil
import unittest
import openml


class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    Curently hard-codes a read-write key.
    Hopefully soon allows using a test server, not the production server.
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
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = "http://test.openml.org/api/v1/xml"
        openml.config.set_cache_directory(self.workdir, self.workdir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir)

__all__ = ['TestBase']
