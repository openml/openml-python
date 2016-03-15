from openml.apiconnector import APIConnector
from openml.autorun import openml_run
from sklearn import ensemble
import xmltodict
import os
"""
An example of an automated machine learning experiment using openml_run
"""

key_file_path = "apikey.txt"
with open(key_file_path, 'r') as fh:
    key = fh.readline()

task_id = 59

clf = ensemble.RandomForestClassifier()
connector = APIConnector(apikey = key)
task = connector.download_task(task_id)

prediction_path, description_path = openml_run(task, clf)

prediction_abspath = os.path.abspath(prediction_path)
description_abspath = os.path.abspath(description_path)

return_code, response = connector.upload_run(prediction_abspath, description_abspath)

if(return_code == 200):
    response_dict = xmltodict.parse(response.content)
    run_id = response_dict['oml:upload_run']['oml:run_id']
    print("Uploaded run with id %s" % (run_id))
