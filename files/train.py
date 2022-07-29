import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

import pickle
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocessing

# load data
data_raw=pd.read_csv("./data/Train.csv")

# preprocess data
data = preprocessing(data_raw, use_location=False, only_means=False)

# split data into X and Y
X = data.drop('target', axis=1)
y = data['target']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# save data
print("Saving data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)

# model
print("Training Extra Trees Regressor")
reg = ExtraTreesRegressor(n_estimators=400, max_depth=50, max_features= 'log2', random_state=42)
reg.fit(X_train, y_train)

#saving the model
print("Saving model in the model folder")
filename = 'models/ExtraTrees_model.sav'
pickle.dump(reg, open(filename, 'wb'))