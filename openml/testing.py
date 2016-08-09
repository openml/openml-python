import inspect
import os
import shutil
import unittest
import openml
import hashlib

class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    Currently hard-codes a read-write key.
    Hopefully soon allows using a test server, not the production server.
    """

    def setUp(self):
        # This cache directory is checked in to git to simulate a populated
        # cache
        self.static_cache_dir = None
        static_cache_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        for i in range(2):
            static_cache_dir = os.path.abspath(os.path.join(static_cache_dir,
                                                            '..'))
            content = os.listdir(static_cache_dir)
            if 'files' in content:
                self.static_cache_dir = os.path.join(static_cache_dir, 'files')
        if self.static_cache_dir is None:
            raise ValueError('Cannot find test cache dir!')

        self.cwd = os.getcwd()
        workdir = os.path.dirname(os.path.abspath(__file__))
        self.workdir = os.path.join(workdir, "tmp")
        try:
            shutil.rmtree(self.workdir)
        except:
            pass

        os.mkdir(self.workdir)
        os.chdir(self.workdir)

        # Remove testmode once mock.wraps is available?
        openml.config._testmode = True
        apikey = openml.config.apikey
        pid = os.getpid()
        md5 = hashlib.md5()
        md5.update(apikey.encode('utf-8'))
        md5.update(str(pid).encode('utf-8'))
        sentinel = md5.hexdigest()
        # For testing the hash code mustn't be bulletproof
        self.sentinel = '%sTESTSENTINEL999' % sentinel[:8]
        openml.config.testsentinel = self.sentinel

        self.cached = True
        # amueller's read/write key that he will throw away later
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = "http://test.openml.org/api/v1/xml"
        openml.config.set_cache_directory(self.workdir, self.workdir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir)

        openml.config._testmode = False

__all__ = ['TestBase']
