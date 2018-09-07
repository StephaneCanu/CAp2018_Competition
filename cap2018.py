#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 18:25:18 2018

@author: stephane
"""

import numpy as np
import pandas as pd

#%% read the data
data = pd.read_csv("./data/train_cap2018.csv", sep=",",header=0)

#%% prepare the data
y = data["level1"].as_matrix()
u, indices = np.unique(y, return_inverse=True)
y = indices+1
X = data.as_matrix(columns=data.columns[2:51])

from sklearn import preprocessing
X = preprocessing.scale(X)

#%% prepare the training

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import time
import datetime as dt

# split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%% train SVM
param_C = 100
param_gamma = 0.0125
classifier = svm.SVC(probability=False,cache_size=2000,C=param_C,gamma=param_gamma)

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
print('Elapsed learning {}'.format(str(dt.datetime.now() - start_time)))

#%%
# Evaluate SVM

predicted = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, predicted)
print("Confusion matrix:\n%s" % cm)

Cost_M =[[ 0, 1 ,    2  ,   3  ,  4  ,   6],
         [ 1, 0 ,    1  ,   4  ,  5  ,   8],
         [ 3, 2 ,    0  ,   3  ,  5  ,   8],
         [10, 7 ,    5  ,   0  ,  2  ,   7],
         [20,16 ,   12  ,   4  ,  0  ,   8],
         [44,38 ,   32  ,  19  , 13  ,   0]]

C = 100*sum(sum(cm*Cost_M))/sum(sum(cm))
print(C)

#%%
def Eval_CAp_2018(y_estimated,y_real):
    """   M is a 6 by 6 confusion matrix real/prediction    
    """
    
    from sklearn import metrics
    M = metrics.confusion_matrix(y_real, y_estimated)
    Cost_M =[[ 0, 1 ,    2  ,   3  ,  4  ,   6],
             [ 1, 0 ,    1  ,   4  ,  5  ,   8],
             [ 3, 2 ,    0  ,   3  ,  5  ,   8],
             [10, 7 ,    5  ,   0  ,  2  ,   7],
             [20,16 ,   12  ,   4  ,  0  ,   8],
             [44,38 ,   32  ,  19  , 13  ,   0]]

    C = 100*sum(sum(M*Cost_M))/sum(sum(M))
    return C , M

#%%
    
C1,cm = Eval_CAp_2018(predicted,y_test)
print(C1)

#%%

x = np.genfromtxt("toto.txt", dtype=None)

C2,cm = Eval_CAp_2018(x[:,0],x[:,1])
print(C2)



