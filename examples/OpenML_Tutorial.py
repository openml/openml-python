"""
OpenML Introduction
===================

A simple tutorial on how to get started on using OpenML.
"""

# coding: utf-8

# # OpenML in Python 
# OpenML is an online collaboration platform for machine learning: 
# 
# * Find or share interesting, well-documented datasets
# * Define research / modelling goals (tasks)
# * Explore large amounts of machine learning algorithms, with APIs in Java, R, Python
# * Log and share reproducible experiments, models, results 
# * Works seamlessly with scikit-learn and other libraries
# * Large scale benchmarking, compare to state of the art

# # Installation
# 
# * Up to now: `pip install git+https://github.com/openml/openml-python.git@develop`
# * In the future: `pip install openml`
# * Check out the installation guide: [https://openml.github.io/openml-python/stable/#installation](https://openml.github.io/openml-python/stable/#installation)

# ## Authentication
# 
# * Create an OpenML account (free) on http://www.openml.org. 
# * After logging in, open your account page (avatar on the top right)
# * Open 'Account Settings', then 'API authentication' to find your API key.
# 
# There are two ways to authenticate:  
# 
# * Create a plain text file `~/.openml/config` with the line 'apikey=MYKEY', replacing MYKEY with your API key.
# * Run the code below, replacing 'YOURKEY' with your API key.

# In[1]:


# Uncomment and set your OpenML key. Don't share your key with others.
import openml as oml
#oml.config.apikey = 'YOURKEY'


# # It all starts with data
# Explore thousands of datasets, or share your own

# ### List datasets

# In[2]:


import openml as oml
openml_list = oml.datasets.list_datasets() # Returns a dict

# Show a nice table with some key data properties
import pandas as pd
datalist = pd.DataFrame.from_dict(openml_list, orient='index') 
datalist = datalist[[
    'did','name','NumberOfInstances',
    'NumberOfFeatures','NumberOfClasses'
]]
print("First 10 of %s datasets..." % len(datalist))
datalist.head(n=10)


# ### Exercise
# - Find datasets with more than 10000 examples
# - Find a dataset called 'eeg_eye_state'
# - Find all datasets with more than 50 classes

# In[3]:


datalist[datalist.NumberOfInstances>10000
        ].sort_values(['NumberOfInstances']).head(n=20)


# In[4]:


datalist.query('name == "eeg-eye-state"')


# In[5]:


datalist.query('NumberOfClasses > 50')


# ## Download datasets
# This is done based on the dataset ID ('did').

# In[6]:


dataset = oml.datasets.get_dataset(1471)

# Print a summary
print("This is dataset '%s', the target feature is '%s'" % 
      (dataset.name, dataset.default_target_attribute))
print("URL: %s" % dataset.url)
print(dataset.description[:500])


# Get the actual data.  
# Returned as numpy array, with meta-info (e.g. target feature, feature names,...)

# In[7]:


X, y, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    return_attribute_names=True,
)
eeg = pd.DataFrame(X, columns=attribute_names)
eeg['class'] = y
print(eeg[:10])


# ### Exercise
# - Explore the data visually

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
eegs = eeg.sample(n=1000)
_ = pd.plotting.scatter_matrix(
    eegs.iloc[:100,:4], 
    c=eegs[:100]['class'], 
    figsize=(10, 10), 
    marker='o', 
    hist_kwds={'bins': 20}, 
    alpha=.8, 
    cmap='plasma'
)


# ## Train machine learning models
# Train a scikit-learn model on the data manually

# In[9]:


from sklearn import neighbors

dataset = oml.datasets.get_dataset(1471)
X, y = dataset.get_data(target=dataset.default_target_attribute)
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)


# You can also ask for meta-data to automatically preprocess the data
# - e.g. categorical features -> do feature encoding

# In[10]:


from sklearn import preprocessing
dataset = oml.datasets.get_dataset(10)
X, y, categorical = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True,
)
print("Categorical features: %s" % categorical)
enc = preprocessing.OneHotEncoder(categorical_features=categorical)
X = enc.fit_transform(X)
clf.fit(X, y)


# # Tasks: set your own goals
# and invite others to work on the same problem  
# Note: tasks are typically created in the web interface

# ## Listing tasks

# In[11]:


task_list = oml.tasks.list_tasks(size=5000) # Get first 5000 tasks

mytasks = pd.DataFrame.from_dict(task_list, orient='index')
mytasks = mytasks[['tid','did','name','task_type','estimation_procedure','evaluation_measures']]
print("First 5 of %s tasks:" % len(mytasks))
mytasks.head()


# ### Exercise
# Search for the tasks on the 'eeg-eye-state' dataset

# In[12]:


mytasks.query('name=="eeg-eye-state"')


# ## Download tasks

# In[13]:


from pprint import pprint
task = oml.tasks.get_task(14951)
pprint(vars(task))


# # Runs: Easily explore models by running them on tasks
# We can run (many) scikit-learn algorithms on (many) OpenML tasks.

# In[14]:


from sklearn import ensemble, tree

# Get a task
task = oml.tasks.get_task(14951)

# Build any classifier or pipeline
clf = tree.ExtraTreeClassifier()

# Create a flow
flow = oml.flows.sklearn_to_flow(clf)

# Run the flow 
run = oml.runs.run_flow_on_task(task, flow)


# Share the run on the OpenML server

# In[15]:


myrun = run.publish()
print("Uploaded to http://www.openml.org/r/" + str(myrun.run_id))


# ### It also works with pipelines
# When you need to handle 'dirty' data, build pipelines to model then automatically

# In[16]:


from sklearn import pipeline, ensemble, preprocessing
from openml import tasks,runs, datasets
task = tasks.get_task(59)
pipe = pipeline.Pipeline(steps=[
            ('Imputer', preprocessing.Imputer(strategy='median')),
            ('OneHotEncoder', preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')),
            ('Classifier', ensemble.RandomForestClassifier())
           ])
flow = oml.flows.sklearn_to_flow(pipe)

run = oml.runs.run_flow_on_task(task, flow)
myrun = run.publish()
print("Uploaded to http://www.openml.org/r/" + str(myrun.run_id))


# ## Download previous results
# You can download all your results anytime, as well as everybody else's  
# List runs by uploader, flow, task, tag, id, ...

# In[17]:


import seaborn as sns
import pandas as pd

# Get the list of runs for task 14951
myruns = oml.runs.list_runs(task=[14951], size=100)

# Download the tasks and plot the scores
scores = []
for id, _ in myruns.items():
    run = oml.runs.get_run(id)
    scores.append({"flow":run.flow_name, "score":run.evaluations['area_under_roc_curve']})
    
sns.violinplot(x="score", y="flow", data=pd.DataFrame(scores), scale="width", palette="Set3");


# ## A Challenge
# Try to build the best possible models on several OpenML tasks, and compare your results with the rest of the class, and learn from them. Some tasks you could try (or browse openml.org):
# 
# * EEG eye state: data_id:[1471](http://www.openml.org/d/1471), task_id:[14951](http://www.openml.org/t/14951)
# * Volcanoes on Venus: data_id:[1527](http://www.openml.org/d/1527), task_id:[10103](http://www.openml.org/t/10103)
# * Walking activity: data_id:[1509](http://www.openml.org/d/1509), task_id: [9945](http://www.openml.org/t/9945), 150k instances
# * Covertype (Satellite): data_id:[150](http://www.openml.org/d/150), task_id: [218](http://www.openml.org/t/218). 500k instances
# * Higgs (Physics): data_id:[23512](http://www.openml.org/d/23512), task_id:[52950](http://www.openml.org/t/52950). 100k instances, missing values

# Easy benchmarking:

# In[18]:


for task_id in [14951, ]: #  Add further tasks. Disclaimer: they might take some time
    task = oml.tasks.get_task(task_id)
    data = oml.datasets.get_dataset(task.dataset_id)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    flow = oml.flows.sklearn_to_flow(clf)
    
    try:
        run = oml.runs.run_flow_on_task(task, flow)
        myrun = run.publish()
        print("kNN on %s: http://www.openml.org/r/%d" % (data.name, myrun.run_id))
    except oml.exceptions.PyOpenMLError as err:
        print("OpenML: {0}".format(err))

