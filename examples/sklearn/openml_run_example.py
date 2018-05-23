"""
OpenML Run Example
==================

An example of an automated machine learning experiment using run_task
"""

from openml.apiconnector import APIConnector
from openml.autorun import run_task
from sklearn import ensemble
import xmltodict
import os


key_file_path = "apikey.txt"
with open(key_file_path, 'r') as fh:
    key = fh.readline()

task_id = 59

clf = ensemble.RandomForestClassifier()
connector = APIConnector(apikey = key)
task = connector.get_task(task_id)

prediction_path, description_path = run_task(task, clf)

prediction_abspath = os.path.abspath(prediction_path)
description_abspath = os.path.abspath(description_path)

return_code, response = connector.upload_run(prediction_abspath, description_abspath)

if(return_code == 200):
    response_dict = xmltodict.parse(response.content)
    run_id = response_dict['oml:upload_run']['oml:run_id']
    print("Uploaded run with id %s" % (run_id))
