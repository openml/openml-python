.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_tasks_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_tasks_tutorial.py:


Tasks
=====

A tutorial on how to list and download tasks.


.. code-block:: default


    import openml
    import pandas as pd
    from pprint import pprint







Tasks are identified by IDs and can be accessed in two different ways:

1. In a list providing basic information on all tasks available on OpenML.
This function will not download the actual tasks, but will instead download
meta data that can be used to filter the tasks and retrieve a set of IDs.
We can filter this list, for example, we can only list tasks having a
special tag or only tasks for a specific target such as
*supervised classification*.

2. A single task by its ID. It contains all meta information, the target
metric, the splits and an iterator which can be used to access the
splits in a useful manner.

Listing tasks
^^^^^^^^^^^^^

We will start by simply listing only *supervised classification* tasks:


.. code-block:: default


    tasks = openml.tasks.list_tasks(task_type_id=1)







**openml.tasks.list_tasks()** returns a dictionary of dictionaries, we convert it into a
`pandas dataframe <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
to have better visualization and easier access:


.. code-block:: default


    tasks = pd.DataFrame.from_dict(tasks, orient='index')
    print(tasks.columns)
    print("First 5 of %s tasks:" % len(tasks))
    pprint(tasks.head())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Index(['tid', 'ttid', 'did', 'name', 'task_type', 'status',
           'estimation_procedure', 'source_data', 'target_feature',
           'MajorityClassSize', 'MaxNominalAttDistinctValues', 'MinorityClassSize',
           'NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances',
           'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues',
           'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures'],
          dtype='object')
    First 5 of 932 tasks:
       tid  ttid  ...  NumberOfNumericFeatures NumberOfSymbolicFeatures
    1    1     1  ...                      6.0                     33.0
    2    2     1  ...                      6.0                     33.0
    3    3     1  ...                      6.0                     33.0
    4    4     1  ...                      6.0                     33.0
    5    5     1  ...                      6.0                     33.0

    [5 rows x 19 columns]


We can filter the list of tasks to only contain datasets with more than
500 samples, but less than 1000 samples:


.. code-block:: default


    filtered_tasks = tasks.query('NumberOfInstances > 500 and NumberOfInstances < 1000')
    print(list(filtered_tasks.index))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 23, 24, 37, 38, 39, 40, 41, 42, 91, 92, 93, 94, 95, 96, 115, 116, 117, 118, 119, 120, 127, 128, 129, 130, 131, 132, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 181, 182, 183, 184, 185, 186, 193, 194, 195, 196, 197, 198, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 307, 308, 309, 310, 311, 312, 379, 380, 381, 382, 383, 384, 391, 392, 393, 394, 395, 396, 433, 434, 435, 436, 437, 438, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 565, 566, 567, 568, 569, 570, 595, 596, 597, 598, 599, 600, 607, 608, 609, 610, 611, 612, 1069, 1072, 1075, 1084, 1088, 1090, 1093, 1094, 1099, 1101, 1103, 1104, 1105, 1107, 1109, 1110, 1111, 1120, 1132, 1134, 1141, 1154, 1155, 1163, 1168, 1232, 1234, 1239, 1240, 1242, 1244, 1245, 1248, 1250, 1251, 1252, 1253, 1257, 1258, 1260, 1261, 1263, 1264, 1265, 1268, 1269, 1270, 1273, 1276, 1279, 1280, 1283, 1288, 1289, 1297, 1304, 1307, 1309, 1310, 1314, 1317, 1320, 1321, 1323, 1327, 1329, 1330, 1334, 1337, 1339, 1341, 1342, 1343, 1346, 1347, 1349, 1353, 1354, 1355, 1358, 1359, 1366, 1367, 1369, 1376, 1377, 1378, 1380, 1382, 1383, 1384, 1387, 1396, 1399, 1402, 1405, 1406, 1410, 1412, 1413, 1419, 1423, 1426, 1428, 1430, 1433, 1437, 1438, 1439, 1440, 1442, 1443, 1447, 1448, 1453, 1456, 1459, 1460, 1462, 1466, 1469, 1471, 1473, 1476, 1477, 1478, 1479, 1484, 1485, 1487, 1490, 1494, 1496, 1497, 1498, 1502, 1503, 1507, 1508, 1509, 1512, 1515, 1518, 1519, 1520, 1523, 1524, 1531, 1535, 1537, 1539, 1541, 1543]



.. code-block:: default


    # Number of tasks
    print(len(filtered_tasks))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    309


Then, we can further restrict the tasks to all have the same resampling strategy:


.. code-block:: default


    filtered_tasks = filtered_tasks.query('estimation_procedure == "10-fold Crossvalidation"')
    print(list(filtered_tasks.index))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1, 19, 37, 91, 115, 127, 145, 151, 181, 193, 205, 211, 217, 229, 241, 247, 253, 307, 379, 391, 433, 511, 517, 565, 595, 607]



.. code-block:: default


    # Number of tasks
    print(len(filtered_tasks))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    26


Resampling strategies can be found on the
`OpenML Website <http://www.openml.org/search?type=measure&q=estimation%20procedure>`_.

Similar to listing tasks by task type, we can list tasks by tags:


.. code-block:: default


    tasks = openml.tasks.list_tasks(tag='OpenML100')
    tasks = pd.DataFrame.from_dict(tasks, orient='index')
    print("First 5 of %s tasks:" % len(tasks))
    pprint(tasks.head())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    First 5 of 100 tasks:
        tid  ttid  ...  NumberOfNumericFeatures NumberOfSymbolicFeatures
    1     1     1  ...                      6.0                     33.0
    7     7     1  ...                      0.0                     37.0
    13   13     1  ...                     16.0                      1.0
    19   19     1  ...                      4.0                      1.0
    25   25     1  ...                    216.0                      1.0

    [5 rows x 19 columns]


Furthermore, we can list tasks based on the dataset id:


.. code-block:: default


    tasks = openml.tasks.list_tasks(data_id=61)
    tasks = pd.DataFrame.from_dict(tasks, orient='index')
    print("First 5 of %s tasks:" % len(tasks))
    pprint(tasks.head())




.. code-block:: pytb

    Traceback (most recent call last):
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 394, in _memory_usage
        out = func()
      File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 382, in __call__
        exec(self.code, self.globals)
      File "/Users/michaelmmeskhi/Documents/GitHub/openml-python/examples/tasks_tutorial.py", line 82, in <module>
        tasks = openml.tasks.list_tasks(data_id=61)
    TypeError: list_tasks() got an unexpected keyword argument 'data_id'




In addition, a size limit and an offset can be applied both separately and simultaneously:


.. code-block:: default


    tasks = openml.tasks.list_tasks(size=10, offset=50)
    tasks = pd.DataFrame.from_dict(tasks, orient='index')
    pprint(tasks)


**OpenML 100**
is a curated list of 100 tasks to start using OpenML. They are all
supervised classification tasks with more than 500 instances and less than 50000
instances per task. To make things easier, the tasks do not contain highly
unbalanced data and sparse data. However, the tasks include missing values and
categorical features. You can find out more about the *OpenML 100* on
`the OpenML benchmarking page <https://www.openml.org/guide/benchmark>`_.

Finally, it is also possible to list all tasks on OpenML with:


.. code-block:: default

    tasks = openml.tasks.list_tasks()
    tasks = pd.DataFrame.from_dict(tasks, orient='index')
    print(len(tasks))


Exercise
########

Search for the tasks on the 'eeg-eye-state' dataset.


.. code-block:: default


    tasks.query('name=="eeg-eye-state"')


Downloading tasks
^^^^^^^^^^^^^^^^^

We provide two functions to download tasks, one which downloads only a
single task by its ID, and one which takes a list of IDs and downloads
all of these tasks:


.. code-block:: default


    task_id = 1
    task = openml.tasks.get_task(task_id)


Properties of the task are stored as member variables:


.. code-block:: default


    pprint(vars(task))


And:


.. code-block:: default


    ids = [1, 2, 19, 97, 403]
    tasks = openml.tasks.get_tasks(ids)
    pprint(tasks[0])


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  5.049 seconds)


.. _sphx_glr_download_examples_tasks_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tasks_tutorial.py <tasks_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tasks_tutorial.ipynb <tasks_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
