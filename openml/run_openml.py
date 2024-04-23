from openml import datasets

# Fetch the dataset with ID 1471 (Iris dataset, as an example)
openml_dataset = datasets.get_dataset(1471, download_data=True, download_qualities=True, download_features_meta_data=True)


# Get the data (X features, y target) from the dataset
X, y, _, _ = openml_dataset.get_data(target=openml_dataset.default_target_attribute)

# Assuming X and y are loaded as per your script
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Explore the data
print(X.head())
print(X.describe())
print(y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a classifier (example: RandomForest)
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

import time
from tqdm import tqdm

for i in tqdm(range(10)):
    time.sleep(0.5)