

#load data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

train = pd.read_csv('merged_train.csv',sep='\t')
#train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
"""
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
"""


def encode(train, test):
	print(train.head())
	print(test.head())
	le = LabelEncoder().fit(train.species) #turn species to no
	labels = le.transform(train.species)   # encode species strings
	classes = list(le.classes_)                    # save column names for submission
	test_ids = test.id # sample id and index no
	#train = train.drop(['species', 'id'], axis=1)
	train = train.drop(['species', 'id'], axis=1)
	test = test.drop(['id'], axis=1)
	
	return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)


for train_index, test_index in sss: #shuffle. only last set is used
#	if (count!=0):
#		print("2 append")
#		X_train, X_test = np.concatenate((X_train, train.values[train_index]), axis=0), np.concatenate((X_test, train.values[test_index]), axis=0)
#		y_train, y_test = np.concatenate((y_train,labels[train_index]), axis=0), np.concatenate((y_test, labels[test_index]), axis=0)
#	else:
	X_train, X_test = train.values[train_index], train.values[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
	

#	X_train, X_test = np.concatenate(X_train, train.values[train_index]), np.concatenate(X_test, train.values[test_index])
#	y_train, y_test = np.concatenate(y_train,labels[train_index]), np.concatenate(y_test, labels[test_index])
print(y_test.shape)

#clf = KNeighborsClassifier(3)
clf=  LinearDiscriminantAnalysis()
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
clf.fit(X_train, y_train)
name = clf.__class__.__name__

print("="*30)
print(name)

print('****Results****')
train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))

train_predictions = clf.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print("Log Loss: {}".format(ll))

log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
log = log.append(log_entry)


