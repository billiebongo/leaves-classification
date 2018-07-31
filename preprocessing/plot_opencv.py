import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#plot l pit to s pit graphs by classes
OPENCV_INPUT_FILENAME='merged_train_sl_pit1.csv'
OPENCV_DATA = pd.read_csv(OPENCV_INPUT_FILENAME)


# Plot graphs of area, l_pit, s_pit, ratio, centroids of all training samples by classes

def encode(train):
	le = LabelEncoder().fit(train.species) #turn species to no
	labels = le.transform(train.species)   # encode species strings
	classes = list(le.classes_)                    # save column names for submission
	train_ids = train.id
	train = train.drop(['species', 'id'], axis=1)
	return train, labels, train_ids,classes

train, labels, train_ids, classes= encode(OPENCV_DATA)

#ratio graph
plt.scatter(x=labels, y=OPENCV_DATA.ratio)
plt.savefig("ratio.png")

#area graph
plt.scatter(x=labels, y=OPENCV_DATA.area)
plt.savefig("area.png")

#l_pit graph
plt.scatter(x=labels, y=OPENCV_DATA.l_pit)
plt.savefig("l_pit.png")

#s_pit graph
plt.scatter(x=labels, y=OPENCV_DATA.s_pit)
plt.savefig("s_pit.png")