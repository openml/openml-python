import xmltodict

from .._api_calls import _perform_api_call

def get_study(study_id):
    xml_string = _perform_api_call("study/" % (study_id))
    result_dict = xmltodict.parse(xml_string)
    
    pass