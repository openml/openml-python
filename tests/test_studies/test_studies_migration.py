# License: BSD 3-Clause
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd
import pytest
import requests

import openml
from openml._api import api_context
from openml.exceptions import OpenMLCacheException
from openml.study import OpenMLStudy
from openml.study import functions as study_functions


@pytest.fixture(scope="function")
def reset_api_to_v1() -> None:
    """Fixture to ensure API is set to V1 for each test."""
    api_context.set_version("v1", strict=False)
    yield
    api_context.set_version("v1", strict=False)


@pytest.fixture(scope="function")
def api_v2() -> None:
    """Fixture to set API to V2 for tests."""
    api_context.set_version("v2", strict=True)
    yield
    api_context.set_version("v1", strict=False)


def test_list_studies_v1(reset_api_to_v1) -> None:
    """Test listing studies using V1 API."""
    studies_df = study_functions.list_studies()
    assert isinstance(studies_df, pd.DataFrame)
    assert not studies_df.empty


def test_study_exists_v1(reset_api_to_v1) -> None:
    """Test study_exists() using V1 API."""
    # Known existing study
    name = "weka.OneR"
    external_version = "Weka_3.9.0_10153"

    exists = study_functions.study_exists(name, external_version)
    assert exists is not False

    # Known non-existing study
    name = "non.existing.study"
    external_version = "0.0.1"

    exists = study_functions.study_exists(name, external_version)
    assert exists is False


def test_get_studies_v1(reset_api_to_v1) -> None:
    """Test get() method returns a valid OpenMLstudy object using V1 API."""
    # Get the study with ID 2 (weka.OneR)
    study_id = 2
    study = study_functions.get_study(study_id)

    assert isinstance(study, OpenMStudy)
    assert study.study_id == study_id
    assert isinstance(study.name, str)
    assert len(study.name) > 0
    assert isinstance(study.external_version, str)


def test_study_publish_v1(reset_api_to_v1) -> None:
    """Test publishing a study using V1 API."""
    from openml_sklearn.extension import SklearnExtension
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    extension = SklearnExtension()
    dt_study = extension.model_to_study(clf)

    # Publish the study
    published_study = dt_study.publish()

    # Verify the published study has an ID
    assert isinstance(published_study, OpenMLstudy)
    assert getattr(published_study, "id", None) is not None


def test_get_studies_v2(api_v2) -> None:
    """Test get() method returns a valid OpenMLstudy object using V2 API."""
    # Get the study with ID 2 (weka.OneR)
    study_id = 2

    # Now get the full study details
    study = study_functions.get_study(study_id)

    # Verify it's an OpenMLstudy with expected attributes
    assert isinstance(study, OpenMLstudy)
    assert study.study_id == study_id
    assert isinstance(study.name, str)
    assert len(study.name) > 0
    assert isinstance(study.external_version, str)


def test_study_exists_v2(api_v2) -> None:
    """Test study_exists() using V2 API."""
    # Known existing study
    name = "weka.OneR"
    external_version = "Weka_3.9.0_10153"

    exists = study_functions.study_exists(name, external_version)
    assert exists != False

    # Known non-existing study
    name = "non.existing.study"
    external_version = "0.0.1"

    exists = study_functions.study_exists(name, external_version)
    assert exists == False
    