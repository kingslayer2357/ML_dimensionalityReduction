# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 03:34:49 2020

@author: kingslayer
"""

#PCA

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv(r"Wine.csv")

X=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_pca=pca.explained_variance_ratio_

#Applying logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#Predicting
y_pred=classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Appling k fold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
m=accuracies.mean()
