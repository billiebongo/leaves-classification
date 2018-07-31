

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from os import listdir
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from PIL import Image

dataDir = '../data/images/'
train = pd.read_csv('../data/'+'train.csv')
image_paths = [dataDir  + f for f in listdir(dataDir)]

#returns image given id
def imageById(id_):
    img = Image.open('../data/images/'+str(id_)+'.jpg')
    #img = img.resize((50, 50), Image.ANTIALIAS)
    return np.array(img)

#returns PCA of image given Id
def imageByIdPCA(id_):
    img = Image.open('../data/images/'+str(id_)+'.jpg')
    img = img.resize((50, 50), Image.ANTIALIAS)
    from sklearn.decomposition import PCA
    # Make an instance of the Model
    pca = PCA(35)
    pcm = pca.fit_transform(img)
    return pcm

#returns PCA data for test and train samples in dimensions (990,35) and (584, 35)
def get_pca_data():
    df = pd.read_csv('../data/train.csv')
    df2 = pd.read_csv('../data/test.csv')
    ids = df.values[:,0].astype(np.int) #ids of train examples
    testids = df2.values[:,0].astype(np.int) #ids of test samples
    pca_train = []
    pca_test = []
    for id_ in ids:
        img=Image.open('../data/images/'+str(id_)+'.jpg')
        img = img.resize((50, 50), Image.ANTIALIAS) #resize to 50x50
        img= np.array(img)
        pca_train.append(img)
    for id_ in testids:
        img=Image.open('../data/images/'+str(id_)+'.jpg')
        img = img.resize((50, 50), Image.ANTIALIAS)
        img= np.array(img)
        pca_test.append(img)
    pca_train=np.array(pca_train)
    pca_test=np.array(pca_test)
    pca_train = pca_train.reshape(990,2500)
    pca_test = pca_test.reshape(594,2500)
    from sklearn.decomposition import PCA
    pca = PCA(35)
    pca.fit(pca_train)
    pca_train = pca.transform(pca_train)
    pca_test = pca.transform(pca_test)
    pca_train=pd.DataFrame(pca_train)
    pca_test = pd.DataFrame(pca_test)
    #print(pca_test.shape)
    #print(pca_train.shape)
    return pca_train, pca_test
