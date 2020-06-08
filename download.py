
import numpy as np
import os
from sklearn import datasets
import pandas as pd


# Load Iris dataset, then join labels and features
iris = datasets.load_iris()
joined_iris = np.insert(iris.data, 0, iris.target, axis=1)

# Create directory and write csv
os.makedirs('./data', exist_ok=True)
np.savetxt('./data/train.csv', joined_iris, delimiter=',', fmt='%1.1f, %1.3f, %1.3f, %1.3f, %1.3f')

df = pd.read_csv('data/train.csv')
X_test = df.iloc[:, 1:]

X_test.to_csv('data/test.csv', index=False)