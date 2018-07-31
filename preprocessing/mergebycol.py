#merge based on column value

import pandas as pd
import numpy as np

#
# Takes output from analysis of openCV2 and merge with test and train
# produces marged_train/ merged test
#
#


OPENCV_INPUT_FILENAME='all_image_output_v2.csv'
DATA_DIR='../'
OPEN_CV_ATTEMPT_NO='1'

train = pd.read_csv(DATA_DIR+'train.csv')
#train_img = pd.read_csv('reorder_train_img.csv')

test = pd.read_csv(DATA_DIR+'test.csv')
test_img = pd.read_csv(OPENCV_INPUT_FILENAME) #assume openCV csv is in the same directory

#create proper pd table for opencv image analysis output
for i in range(1584): #convert 321.jpg to 321 and convert string to int64 type
	test_img['id'].iloc[i]=np.int64(test_img['id'].iloc[i][:-4])
	print(test_img['id'].iloc[i])
print(test_img.tail())
#rename column to img
test_img.columns = ['def_count', 'id', 'l_pit','moments_x', 'moments_y','ratio', 's_pit', 'wxh']
print(test_img.head())
print(type(test_img['id'].iloc[9]))
print(type(test['id'].iloc[9]))
print(train.shape)
print(test.shape)
print(test_img.shape)
merged_train=pd.merge(train, test_img, on='id')
merged_test=pd.merge(test, test_img, on='id')
print(merged_train.head())
print(len(merged_train)) #shld be 594
merged_train.to_csv('merged_train_sl_pit{}.csv'.format(OPEN_CV_ATTEMPT_NO), sep='\t')
merged_test.to_csv('merged_test_sl_pit{}.csv'.format(OPEN_CV_ATTEMPT_NO), sep='\t')
