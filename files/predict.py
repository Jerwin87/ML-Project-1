import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

# set paths
model = 'models/ExtraTrees_model.sav'
X_test_path = "data/X_test.csv"
y_test_path = "data/y_test.csv"
X_train_path = "data/X_train.csv"
y_train_path = "data/y_train.csv"

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)
X_train= pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

# predict on train set
y_pred_train = loaded_model.predict(X_train)
print('RMSE on train set: ', mean_squared_error(y_train, y_pred_train, squared=False))

#predict on test set
y_pred_test = loaded_model.predict(X_test)
print('RMSE on test set: ', mean_squared_error(y_test, y_pred_test, squared=False))

# convert to binary classification
y_pred_C = (y_pred_test>56).astype(int)
y_test_C = (y_test>56).astype(int)
print('confusion matrix: ')
print(confusion_matrix(y_test_C, y_pred_C))
print('accuracy: ', accuracy_score(y_test_C, y_pred_C))
print('recall: ', recall_score(y_test_C, y_pred_C))
print('precision: ', precision_score(y_test_C, y_pred_C))
