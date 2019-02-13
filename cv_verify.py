from sklearn.model_selection import cross_val_score
import sklearn.svm
import pandas as pd
import numpy as np

test_x = pd.read_csv("data/test_x.csv")
test_y = pd.read_csv("data/test_y_1col.csv")
train_x = pd.read_csv("data/train_x.csv")
train_y = pd.read_csv("data/train_y_1col.csv")
test_y = np.ravel(test_y)
train_y = np.ravel(train_y)

clf = sklearn.svm.SVC(kernel='rbf', C=13, gamma=1)

print("start train set cv")
train_scores = cross_val_score(clf, train_x, train_y, cv=3)
print(train_scores)
print(train_scores.mean())

print("start test set cv")
test_scores = cross_val_score(clf, test_x, test_y, cv=3)
print(test_scores)
print(test_scores.mean())

