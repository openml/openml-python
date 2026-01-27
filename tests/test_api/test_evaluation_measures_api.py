# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
from unittest.mock import Mock 
  
import requests  
  
from openml._api.resources.evaluation_measures import (  
    EvaluationMeasuresV1,  
    EvaluationMeasuresV2,  
)  
from openml.testing import TestBase  
  
  
class TestEvaluationMeasuresV1(TestBase):  
    """Tests for V1 XML API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:  
        """Set up test fixtures."""  
        super().setUp() 
        self.mock_http = Mock()  
        self.api = EvaluationMeasuresV1(http=self.mock_http)  
  
    def test_list_evaluation_measures_success(self):  
        """Test successful listing of evaluation measures from V1 API."""  
        xml_response = """<oml:evaluation_measures xmlns:oml="http://openml.org/openml">  
<oml:measures>  
<oml:measure>area_under_roc_curve</oml:measure>  
<oml:measure>average_cost</oml:measure>  
<oml:measure>f_measure</oml:measure>  
<oml:measure>predictive_accuracy</oml:measure>  
<oml:measure>precision</oml:measure>  
<oml:measure>recall</oml:measure>  
</oml:measures>  
</oml:evaluation_measures>"""  
  
        mock_response = Mock(spec=requests.Response)  
        mock_response.text = xml_response  
        self.mock_http.get.return_value = mock_response  
  
        measures = self.api.list()  
  
        self.mock_http.get.assert_called_once_with("evaluationmeasure/list")  
        assert isinstance(measures, list)  
        assert len(measures) == 6  
        assert "predictive_accuracy" in measures  
        assert "area_under_roc_curve" in measures  
  
    def test_list_missing_root_element(self):  
        """Test error when XML response is missing root element."""  
        xml_response = """<oml:invalid_root xmlns:oml="http://openml.org/openml">  
<oml:measures>  
<oml:measure>predictive_accuracy</oml:measure>  
</oml:measures>  
</oml:invalid_root>"""  
  
        mock_response = Mock(spec=requests.Response)  
        mock_response.text = xml_response  
        self.mock_http.get.return_value = mock_response  
  
        with pytest.raises(ValueError) as excinfo:  
            self.api.list()  
  
        assert 'does not contain "oml:evaluation_measures"' in str(excinfo.value)  
  
  
class TestEvaluationMeasuresV2(TestBase):  
    """Tests for V2 JSON API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:  
        """Set up test fixtures."""  
        super().setUp()  
        self.mock_http = Mock()  
        self.api = EvaluationMeasuresV2(http=self.mock_http)  
  
    def test_list_evaluation_measures_success(self):  
        """Test successful listing of evaluation measures from V2 API."""  
        json_response = [  
            "area_under_roc_curve",  
            "average_cost",  
            "f_measure",  
            "predictive_accuracy",  
            "precision",  
            "recall",  
        ]  
  
        mock_response = Mock(spec=requests.Response)  
        mock_response.json.return_value = json_response  
        self.mock_http.get.return_value = mock_response  
  
        measures = self.api.list()  
  
        self.mock_http.get.assert_called_once_with("evaluationmeasure/list")  
        assert isinstance(measures, list)  
        assert len(measures) == 6  
        assert "predictive_accuracy" in measures  
        assert "area_under_roc_curve" in measures  
  
    def test_list_invalid_type(self):  
        """Test error when response is not a list."""  
        json_response = {  
            "measures": ["predictive_accuracy", "f_measure"],  
        }  
  
        mock_response = Mock(spec=requests.Response)  
        mock_response.json.return_value = json_response  
        self.mock_http.get.return_value = mock_response  
  
        with pytest.raises(ValueError) as excinfo:  
            self.api.list()  
  
        assert "Expected list, got" in str(excinfo.value)  
  