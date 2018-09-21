"""
Dataset upload tutorial
=======================

A tutorial on how to create and upload a dataset to OpenML.
"""
import numpy as np
import pandas as pd
import openml
import sklearn.datasets

############################################################################
# For this example we will upload to the test server to not  pollute the live
# server with countless copies of the same dataset.
openml.config.server = 'https://test.openml.org/api/v1/xml'

############################################################################
# Uploading a data set store in a NumPy array
############################################################################

############################################################################
# Prepare the data
# ^^^^^^^^^^^^^^^^
# Load an example dataset from scikit-learn which we will upload to OpenML.org
# via the API.
breast_cancer = sklearn.datasets.load_breast_cancer()
name = 'BreastCancer(scikit-learn)'
X = breast_cancer.data
y = breast_cancer.target
attribute_names = breast_cancer.feature_names
targets = breast_cancer.target_names
description = breast_cancer.DESCR

############################################################################
# OpenML does not distinguish between the attributes and targets on the data
# level and stores all data in a single matrix. The target feature is indicated
# as meta-data of the dataset (and tasks on that data).
data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
attribute_names = list(attribute_names)
attributes = [
    (attribute_name, 'REAL') for attribute_name in attribute_names
] + [('class', 'REAL')]

############################################################################
# Create the dataset object
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# The definition of all fields can be found in the XSD files describing the
# expected format:
#
# https://github.com/openml/OpenML/blob/master/openml_OS/views/pages/api_new/v1/xsd/openml.data.upload.xsd
dataset = openml.datasets.functions.create_dataset(
    # The name of the dataset (needs to be unique).
    # Must not be longer than 128 characters and only contain
    # a-z, A-Z, 0-9 and the following special characters: _\-\.(),
    name=name,
    # Textual description of the dataset.
    description=description,
    # The person who created the dataset.
    creator='Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian',
    # People who contributed to the current version of the dataset.
    contributor=None,
    # The date the data was originally collected, given by the uploader.
    collection_date='01-11-1995',
    # Language in which the data is represented.
    # Starts with 1 upper case letter, rest lower case, e.g. 'English'.
    language='English',
    # License under which the data is/will be distributed.
    licence='BSD (from scikit-learn)',
    # Name of the target. Can also have multiple values (comma-separated).
    default_target_attribute='class',
    # The attribute that represents the row-id column, if present in the
    # dataset.
    row_id_attribute=None,
    # Attributes that should be excluded in modelling, such as identifiers and
    # indexes.
    ignore_attribute=None,
    # How to cite the paper.
    citation=(
        "W.N. Street, W.H. Wolberg and O.L. Mangasarian. "
        "Nuclear feature extraction for breast tumor diagnosis. "
        "IS&T/SPIE 1993 International Symposium on Electronic Imaging: "
        "Science and Technology, volume 1905, pages 861-870, "
        "San Jose, CA, 1993."
    ),
    # Attributes of the data
    attributes=attributes,
    data=data,
    # Format of the dataset. Only 'arff' for now.
    format='arff',
    # A version label which is provided by the user.
    version_label='test',
    original_data_url=('https://archive.ics.uci.edu/ml/datasets/Breast+Cancer'
                       '+Wisconsin+(Diagnostic)'),
    paper_url=('https://www.spiedigitallibrary.org/conference-proceedings-of'
               '-spie/1905/0000/Nuclear-feature-extraction-for-breast-tumor-'
               'diagnosis/10.1117/12.148698.short?SSO=1')
)

############################################################################
try:
    upload_id = dataset.publish()
    print('URL for dataset: %s/data/%d' % (openml.config.server, upload_id))
except openml.exceptions.PyOpenMLError as err:
    print("OpenML: {0}".format(err))

############################################################################
# Uploading a dataset stored in a Pandas DataFrame
############################################################################

############################################################################
# I might happen that your dataset is made of heterogeneous data which can be
# usually stored as a Pandas DataFrame. DataFrame offers the adavantages to
# store the type of data for each column as well as the attribute names.
# Therefore, when providing a Pandas DataFrame, OpenML can infer those
# information without the need to specifically provide them when calling the
# function :func:`create_dataset`. In this regard, you only need to pass
# ``'auto'`` to the ``attributes`` parameter.

############################################################################
# Create a fake minimalist dataset stored inside a dataframe.

data = [
    ['a', 'sunny', 85.0, 85.0, 'FALSE', 'no'],
    ['b', 'sunny', 80.0, 90.0, 'TRUE', 'no'],
    ['c', 'overcast', 83.0, 86.0, 'FALSE', 'yes'],
    ['d', 'rainy', 70.0, 96.0, 'FALSE', 'yes'],
    ['e', 'rainy', 68.0, 80.0, 'FALSE', 'yes'],
    ['f', 'rainy', 65.0, 70.0, 'TRUE', 'no'],
    ['g', 'overcast', 64.0, 65.0, 'TRUE', 'yes'],
    ['h', 'sunny', 72.0, 95.0, 'FALSE', 'no'],
    ['i', 'sunny', 69.0, 70.0, 'FALSE', 'yes'],
    ['j', 'rainy', 75.0, 80.0, 'FALSE', 'yes'],
    ['k', 'sunny', 75.0, 70.0, 'TRUE', 'yes'],
    ['l', 'overcast', 72.0, 90.0, 'TRUE', 'yes'],
    ['m', 'overcast', 81.0, 75.0, 'FALSE', 'yes'],
    ['n', 'rainy', 71.0, 91.0, 'TRUE', 'no']
]
column_names = ['rnd_str', 'outlook', 'temperature', 'humidity',
            'windy', 'play']
df = pd.DataFrame(data, columns=column_names)
# enforce the categorical column to have a categorical dtype
df['outlook'] = df['outlook'].astype('category')
df['windy'] = df['windy'].astype('category')
df['play'] = df['play'].astype('category')
print(df.info())

############################################################################
# We enforce the column 'outlook', 'winday', and 'play' to be a categorical
# dtype while the column 'rnd_str' is kept as a string column. Then, we can
# call :func:`create_dataset` by passing the dataframe and fixing the parameter
# ``attributes`` to ``'auto'``.

# force OpenML to infer the attributes from the dataframe
attributes = 'auto'
# meta-information
name = 'Pandas_testing_dataset'
description = 'Synthetic dataset created from a Pandas DataFrame'
creator = 'OpenML tester'
collection_date = '01-01-2018'
language = 'English'
licence = 'MIT'
default_target_attribute = 'play'
citation = 'None'
original_data_url = 'http://openml.github.io/openml-python'
paper_url = 'http://openml.github.io/openml-python'
dataset = openml.datasets.functions.create_dataset(
    name=name,
    description=description,
    creator=creator,
    contributor=None,
    collection_date=collection_date,
    language=language,
    licence=licence,
    default_target_attribute=default_target_attribute,
    row_id_attribute=None,
    ignore_attribute=None,
    citation=citation,
    attributes=attributes,
    data=df,
    format='arff',
    version_label='test',
    original_data_url=original_data_url,
    paper_url=paper_url
)

############################################################################
try:
    upload_id = dataset.publish()
    print('URL for dataset: %s/data/%d' % (openml.config.server, upload_id))
except openml.exceptions.PyOpenMLError as err:
    print("OpenML: {0}".format(err))