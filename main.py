# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 19:02:21 2018

@author: Nikhil Anand
"""
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing CSV file
dataframe=pd.read_csv('Absenteeism_at_work.csv',sep=';')
X = dataframe.iloc[:,:-1]

from sklearn import preprocessing 
nor = preprocessing.normalize(X)#normalize
sta = preprocessing.scale(X)#Standardise



#Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 20, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_train)



