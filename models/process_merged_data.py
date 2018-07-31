#merge train csvs and run random forest with 0.8 validation
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

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
#train_merged=pd.train = pd.read_csv('merged_train_2.csv')
train_merged=pd.train = pd.read_csv('../preprocessing/merged_train_sl_pit1.csv')
test_merged=pd.train = pd.read_csv('../preprocessing/merged_test_sl_pit1.csv')


#
# Run LDA/RF on 64x3 attributes + Opencv2 output
# Results were okkkkkkk. Cross-validation results were pretty high for LDA but test results score were hmm
# Post-mortem: get better processed results from images: PCA and depths of pits. perhaps find out what LDA is?
# 
#

def encode(train, test):
	le = LabelEncoder().fit(train.species) #turn species to no
	labels = le.transform(train.species)   # encode species strings
	classes = list(le.classes_)                    # save column names for submission
	test_ids = test.id # sample id and index no
	#train = train.drop(['species', 'id'], axis=1)
	train = train.drop(['species', 'id'], axis=1)
	test = test.drop(['id'], axis=1)
	
	return train, labels, test, test_ids, classes


train, labels, test, test_ids, classes = encode(train, test)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.05, random_state=24)
print(classes)
print(classes[29])
print(classes[37])
print(test.head())

for train_index, test_index in sss: #shuffle. only last set is used
	X_train, X_test = train.values[train_index], train.values[test_index]
	y_train, y_test = labels[train_index], labels[test_index]

# above used for CV only

#clf = KNeighborsClassifier(5)
#clf=LinearDiscriminantAnalysis()
clf=RandomForestClassifier()
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

print("predicting test csv")
test_predictions = clf.predict(test)
print(test_predictions)
test_predictions = clf.predict_proba(test)
print(test_predictions)

submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

#Export Submission
submission.to_csv('submission_1_0.csv', index = False) 
#print(submission.head())
