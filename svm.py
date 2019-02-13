import pandas as pd
import sklearn.svm
import numpy as np
from sklearn.model_selection import GridSearchCV
import datetime
import time

test_x = pd.read_csv("data/test_x.csv")
test_y = pd.read_csv("data/test_y_1col.csv")
train_x = pd.read_csv("data/train_x.csv")
train_y = pd.read_csv("data/train_y_1col.csv")
test_y = np.ravel(test_y)
train_y = np.ravel(train_y)
starttime = datetime.datetime.now()
gd_model = sklearn.svm.SVC(kernel='rbf', decision_function_shape='ovo', cache_size=512)
print(test_x.shape)
print(test_y.shape)
print(train_x.shape)
print(train_y.shape)
c_values = np.linspace(1,20,20)
c_values = c_values.tolist()
gamma_values = np.linspace(0.01, 0.1, 10)
gamma_values = gamma_values.tolist()
gamma_values = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]
print(c_values)
print(gamma_values)
gd_param_grid = {'C' : c_values, 'gamma': gamma_values}
clf = GridSearchCV(gd_model, param_grid=gd_param_grid)
clf.fit(train_x, train_y)
print(clf.best_score_)
print(clf.best_params_)
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
print("OK, baby")

