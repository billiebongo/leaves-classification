#merge train csvs and run random forest with 0.8 validation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pca_leaves import get_pca_data
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
from thresholding import threshold
train_merged=pd.train = pd.read_csv('../preprocessing/merged_train_sl_pit1.csv')
test_merged=pd.train = pd.read_csv('../preprocessing/merged_test_sl_pit1.csv')


#
# Run LDA/RF on 64x3 attributes + Opencv2 output
# Results were okkkkkkk. Cross-validation results were pretty high for LDA but test results score were hmm
# Post-mortem: get better processed results from images: PCA and depths of pits. perhaps find out what LDA is?
# Reminder: change submission file name.
#

def find_low_certainty(test_ids,predict_class, predict_proba):
	for i in range(len(predict_proba)):
		if predict_proba[i][int(predict_class[i])]<0.85:
			print("predicted"+ str(test_ids[i])+ "as"+str(predict_class[i])+" with "+str(predict_proba[i][int(predict_class[i])]))

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
#sss = StratifiedShuffleSplit(labels, 10, test_size=0.01, random_state=24)


#for train_index, test_index in sss: #shuffle. only last set is used

#	X_train, X_test = train.values[train_index], train.values[test_index]
#	y_train, y_test = labels[train_index], labels[test_index]


#clf = KNeighborsClassifier(5)
#clf=LinearDiscriminantAnalysis()

pca_train, pca_test = get_pca_data() #dataframe
X_train=pd.concat([train, pca_train], axis=1) #dataframe
y_train=labels #nummpyarray
print(len(y_train))
print(type(y_train))
print(type(X_train))
#X_train['classes']=y_train
x_test=pd.concat([test, pca_test], axis=1)
#X_train.to_csv('PCA_LABELS.csv', index = False)
#X_train=train
#x_test=test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
x_test = scaler.transform(x_test)

RF_clf=RandomForestClassifier()

params = {'C':[1, 10, 50, 100, 500, 1000, 2200, 2500], 'tol': [0.001, 0.0001, 0.005]}
#try newton-cg
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)

RF_clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
name = clf.__class__.__name__

print("="*30)
print(name)


print("predicting test csv")
test_predictions_class = clf.predict(x_test)

test_predictions = clf.predict_proba(x_test)
RF_proba = RF_clf.predict_proba(x_test)
print(test_predictions)
low_certainty_list=find_low_certainty(test_ids,test_predictions_class, test_predictions)
submission = pd.DataFrame(test_predictions, columns=classes)

test_predictions_threshold=threshold(test_predictions, classes, RF_proba)
print("after threshold")
low_certainty_list=find_low_certainty(test_ids,test_predictions_class, test_predictions)
submission_threshold = pd.DataFrame(test_predictions_threshold, columns=classes)
submission_threshold.insert(0, 'id', test_ids)
submission_threshold.reset_index()

#Export Submission
submission_threshold.to_csv('submission_10_0test.csv', index = False)
#4.0 is with PCA AND STANDARDIZATION AND LOGs
#6.0 is with thresholding
#6.0 test is testing the thresholding
#8 is thresholding those >78%, 0.013 woo
#9 is RF_clf