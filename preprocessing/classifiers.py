

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
from models.pca_leaves import get_pca_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
train = pd.read_csv('merged_train_sl_pit1.csv')
test = pd.read_csv('merged_test_sl_pit1.csv')
pca_train, pca_test = get_pca_data()
train=pd.concat([train, pca_train], axis=1)

###############
#
# Retrieve image data by OpenCV, PCA and numerical training data,
# Asses with common out-of-the-box algorithms
# assessed with cross-validation (0.1 and 0.2 )
# And churn out cross-validation accuracies to get an idea of which models could be used
#
###############

classifiers = [
	linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg'),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
	]


#dropping sets of each attribute to figure out relative importance of texture, margin, shape. Results: shape seems carry the most weight.

drop_texture_list=['species', 'id','texture1', 'texture2', 'texture3', 'texture4', 'texture5', 'texture6', 'texture7', 'texture8', 'texture9', 'texture10', 'texture11', 'texture12', 'texture13', 'texture14', 'texture15', 'texture16', 'texture17', 'texture18', 'texture19', 'texture20', 'texture21', 'texture22', 'texture23', 'texture24', 'texture25', 'texture26', 'texture27', 'texture28', 'texture29', 'texture30', 'texture31', 'texture32', 'texture33', 'texture34', 'texture35', 'texture36', 'texture37', 'texture38', 'texture39', 'texture40', 'texture41', 'texture42', 'texture43', 'texture44', 'texture45', 'texture46', 'texture47', 'texture48', 'texture49', 'texture50', 'texture51', 'texture52', 'texture53', 'texture54', 'texture55', 'texture56', 'texture57', 'texture58', 'texture59', 'texture60', 'texture61', 'texture62', 'texture63', 'texture64']
drop_m_and_shape=['species', 'id','margin1', 'shape1', 'margin2', 'shape2', 'margin3', 'shape3', 'margin4', 'shape4', 'margin5', 'shape5', 'margin6', 'shape6', 'margin7', 'shape7', 'margin8', 'shape8', 'margin9', 'shape9', 'margin10', 'shape10', 'margin11', 'shape11', 'margin12', 'shape12', 'margin13', 'shape13', 'margin14', 'shape14', 'margin15', 'shape15', 'margin16', 'shape16', 'margin17', 'shape17', 'margin18', 'shape18', 'margin19', 'shape19', 'margin20', 'shape20', 'margin21', 'shape21', 'margin22', 'shape22', 'margin23', 'shape23', 'margin24', 'shape24', 'margin25', 'shape25', 'margin26', 'shape26', 'margin27', 'shape27', 'margin28', 'shape28', 'margin29', 'shape29', 'margin30', 'shape30', 'margin31', 'shape31', 'margin32', 'shape32', 'margin33', 'shape33', 'margin34', 'shape34', 'margin35', 'shape35', 'margin36', 'shape36', 'margin37', 'shape37', 'margin38', 'shape38', 'margin39', 'shape39', 'margin40', 'shape40', 'margin41', 'shape41', 'margin42', 'shape42', 'margin43', 'shape43', 'margin44', 'shape44', 'margin45', 'shape45', 'margin46', 'shape46', 'margin47', 'shape47', 'margin48', 'shape48', 'margin49', 'shape49', 'margin50', 'shape50', 'margin51', 'shape51', 'margin52', 'shape52', 'margin53', 'shape53', 'margin54', 'shape54', 'margin55', 'shape55', 'margin56', 'shape56', 'margin57', 'shape57', 'margin58', 'shape58', 'margin59', 'shape59', 'margin60', 'shape60', 'margin61', 'shape61', 'margin62', 'shape62', 'margin63', 'shape63', 'margin64', 'shape64']
drop_shape_list=['species', 'id','shape1', 'shape2', 'shape3', 'shape4', 'shape5', 'shape6', 'shape7', 'shape8', 'shape9', 'shape10', 'shape11', 'shape12', 'shape13', 'shape14', 'shape15', 'shape16', 'shape17', 'shape18', 'shape19', 'shape20', 'shape21', 'shape22', 'shape23', 'shape24', 'shape25', 'shape26', 'shape27', 'shape28', 'shape29', 'shape30', 'shape31', 'shape32', 'shape33', 'shape34', 'shape35', 'shape36', 'shape37', 'shape38', 'shape39', 'shape40', 'shape41', 'shape42', 'shape43', 'shape44', 'shape45', 'shape46', 'shape47', 'shape48', 'shape49', 'shape50', 'shape51', 'shape52', 'shape53', 'shape54', 'shape55', 'shape56', 'shape57', 'shape58', 'shape59', 'shape60', 'shape61', 'shape62', 'shape63', 'shape64']
drop_margin_list=['species', 'id','margin1', 'margin2', 'margin3', 'margin4', 'margin5', 'margin6', 'margin7', 'margin8', 'margin9', 'margin10', 'margin11', 'margin12', 'margin13', 'margin14', 'margin15', 'margin16', 'margin17', 'margin18', 'margin19', 'margin20', 'margin21', 'margin22', 'margin23', 'margin24', 'margin25', 'margin26', 'margin27', 'margin28', 'margin29', 'margin30', 'margin31', 'margin32', 'margin33', 'margin34', 'margin35', 'margin36', 'margin37', 'margin38', 'margin39', 'margin40', 'margin41', 'margin42', 'margin43', 'margin44', 'margin45', 'margin46', 'margin47', 'margin48', 'margin49', 'margin50', 'margin51', 'margin52', 'margin53', 'margin54', 'margin55', 'margin56', 'margin57', 'margin58', 'margin59', 'margin60', 'margin61', 'margin62', 'margin63', 'margin64']

# given ground truth values and predictions by crossvalidations: find the classes commonly misclassified
def find_misclassifications_rate(y_test, predictions): #of cross-validation
	error_count={}
	for r in range(99):
		error_count[str(r)]=0
	for i in range(99):
		if (y_test[i]==predictions[i]):
			print("matched {} correctly".format(y_test[i]))
		else:
			print("misclassified {} as {}".format(y_test[i], predictions[i]))
			error_count[str(y_test[i])]+=1
	print(error_count)


def encode(train, test):
	le = LabelEncoder().fit(train.species) # turn species to no
	labels = le.transform(train.species)   # encode species strings
	classes = list(le.classes_)            # save column names for submission
	test_ids = test.id 		       # sample id and index no
	#train = train.drop(['species', 'id'], axis=1)
	train = train.drop(['species', 'id'], axis=1)
	test = test.drop(['id'], axis=1)
	
	return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)

#cross-validation: 0.2 or 0.1 works well enough
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for train_index, test_index in sss: #shuffle - only last set is used.

	X_train, X_test = train.values[train_index], train.values[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
	

#	X_train, X_test = np.concatenate(X_train, train.values[train_index]), np.concatenate(X_test, train.values[test_index])
#	y_train, y_test = np.concatenate(y_train,labels[train_index]), np.concatenate(y_test, labels[test_index])
print(y_test.shape)

for clf in classifiers:
	scaler = StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	clf.fit(X_train, y_train)
	name = clf.__class__.__name__

	print("=" * 30)
	print(name)

	print('Results:')
	train_predictions = clf.predict(X_test)
	acc = accuracy_score(y_test, train_predictions)
	print("Accuracy: {:.4%}".format(acc))

	train_predictions_prob = clf.predict_proba(X_test)
	ll = log_loss(y_test, train_predictions_prob)
	print("Log Loss: {}".format(ll))

	log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
	log = log.append(log_entry)
	find_misclassifications_rate(y_test, train_predictions)


print("=" * 30)

