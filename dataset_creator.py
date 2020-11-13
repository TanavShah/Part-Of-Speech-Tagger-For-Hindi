import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

col_list = ["tokens", "tags"]
df = pd.read_csv("token-tag_pairs_cleaned.csv", usecols=col_list)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

X_train.to_csv (r'train_set.csv', index = False, header=True)
X_test.to_csv (r'test_set.csv', index = False, header=True)