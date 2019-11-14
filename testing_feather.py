from openml import datasets
import time
import pandas as pd

pd.set_option("display.max_colwidth", -1)
def run_get(filename):
    list_datasets = datasets.list_datasets(output_format="dataframe")
    first_time = []
    ids = list_datasets
    for did in ids['did']:
        start = time.time()
        try:
            data = datasets.get_dataset(did)
            data.get_data()
        except:
            pass
        end = time.time()
        first_time.append(end-start)


    ids["first_time"] = first_time
    ids.to_pickle(filename)

#run_get("list_second.pkl")
df1 = pd.read_pickle("list.pkl")
df2 = pd.read_pickle("list_second.pkl")
print(len(df1), len(df2))
compare = df1["first_time"] < df2["first_time"]
print(compare[compare].index)
df1.to_csv('df1.csv')
df2.to_csv('df2.csv')