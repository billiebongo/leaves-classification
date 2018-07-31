#classify the images

#match leave name to image id
#
from sklearn.preprocessing import LabelEncoder
#return all the image files when given id
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



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#generate shell script to delete
#chmod +x sort_img.sh
#sudo chown -R $USER: $HOME
def return_image_filenames_per_species_no(species_no, image_ids, labels):
	for i in range(labels.shape[0]):
		if labels[i]==species_no:
			print("cp {}.jpg order_img/{}".format(image_ids[i],species_no))

def delete_train_img(image_ids):
	for i in range(len(image_ids)):
		print("rm {}.jpg".format(image_ids[i]))
		



def encode(train, test):
	le = LabelEncoder().fit(train.species) #turn species to no
	labels = le.transform(train.species)   # their species no
	classes = list(le.classes_)     #99 unique species
	test_ids = test.id # sample id and index no
	#train = train.drop(['species', 'id'], axis=1)
	image_ids=train['id']
	train = train.drop(['species', 'id'], axis=1)
	test = test.drop(['id'], axis=1)
	
	return train, labels, test, test_ids, classes, image_ids

train, labels, test, test_ids, classes, image_ids = encode(train, test)
delete_train_img(image_ids)
#for s in range(99):
#	return_image_filenames_per_species_no(s, image_ids, labels)
