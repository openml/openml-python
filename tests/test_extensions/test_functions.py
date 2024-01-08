# License: BSD 3-Clause
from __future__ import annotations

import inspect

import pytest

import openml.testing
from openml.extensions import get_extension_by_flow, get_extension_by_model, register_extension


class DummyFlow:
    external_version = "DummyFlow==0.1"
    dependencies = None


class DummyModel:
    pass


class DummyExtension1:
    @staticmethod
    def can_handle_flow(flow):
        return inspect.stack()[2].filename.endswith("test_functions.py")

    @staticmethod
    def can_handle_model(model):
        return inspect.stack()[2].filename.endswith("test_functions.py")


class DummyExtension2:
    @staticmethod
    def can_handle_flow(flow):
        return False

    @staticmethod
    def can_handle_model(model):
        return False


def _unregister():
    # "Un-register" the test extensions
    while True:
        rem_dum_ext1 = False
        rem_dum_ext2 = False
        try:
            openml.extensions.extensions.remove(DummyExtension1)
            rem_dum_ext1 = True
        except ValueError:
            pass
        try:
            openml.extensions.extensions.remove(DummyExtension2)
            rem_dum_ext2 = True
        except ValueError:
            pass
        if not rem_dum_ext1 and not rem_dum_ext2:
            break


class TestInit(openml.testing.TestBase):
    def setUp(self):
        super().setUp()
        _unregister()

    def test_get_extension_by_flow(self):
        assert get_extension_by_flow(DummyFlow()) is None
        with pytest.raises(ValueError, match="No extension registered which can handle flow:"):
            get_extension_by_flow(DummyFlow(), raise_if_no_extension=True)
        register_extension(DummyExtension1)
        assert isinstance(get_extension_by_flow(DummyFlow()), DummyExtension1)
        register_extension(DummyExtension2)
        assert isinstance(get_extension_by_flow(DummyFlow()), DummyExtension1)
        register_extension(DummyExtension1)
        with pytest.raises(
            ValueError, match="Multiple extensions registered which can handle flow:"
        ):
            get_extension_by_flow(DummyFlow())

    def test_get_extension_by_model(self):
        assert get_extension_by_model(DummyModel()) is None
        with pytest.raises(ValueError, match="No extension registered which can handle model:"):
            get_extension_by_model(DummyModel(), raise_if_no_extension=True)
        register_extension(DummyExtension1)
        assert isinstance(get_extension_by_model(DummyModel()), DummyExtension1)
        register_extension(DummyExtension2)
        assert isinstance(get_extension_by_model(DummyModel()), DummyExtension1)
        register_extension(DummyExtension1)
        with pytest.raises(
            ValueError, match="Multiple extensions registered which can handle model:"
        ):
            get_extension_by_model(DummyModel())
