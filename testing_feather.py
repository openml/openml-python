from openml import datasets
import time
import  pandas as pd
list_datasets = datasets.list_datasets(output_format="dataframe")
first_time = []
ids = list_datasets
for did in ids['did']:
    start = time.time()
    data = datasets.get_dataset(did)
    data.get_data()
    end = time.time()
    first_time.append(end-start)

pd.set_option("display.max_colwidth", -1)
ids["first_time"] = first_time
ids.to_pickle("list.pkl")