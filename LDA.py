# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 04:03:00 2020

@author: kingslayer
"""

class AverageAccuracy():
    average_accuracy=0
    average=0
    s=0
    total=0
    def __init__(self,average_number):
        self.average_number=average_number
    def work(self):
            
        #importing the libraries
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            
           
            
            for i in range(0,self.average_number):
                 #importing the dataset
                dataset=pd.read_csv(r"Wine.csv")
            
                X=dataset.iloc[:,0:13].values
                y=dataset.iloc[:,13].values
                
                #splitting into training and test set
                from sklearn.model_selection import train_test_split
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
                
                #Feature scaling
                from sklearn.preprocessing import StandardScaler
                sc_X=StandardScaler()
                X_train=sc_X.fit_transform(X_train)
                X_test=sc_X.transform(X_test)
                
                #Applying LDA
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
                lda=LDA(n_components=3)
                X_train=lda.fit_transform(X_train,y_train)
                X_test=lda.transform(X_test)
                
                
                #Applying logistic regression
                from sklearn.linear_model import LogisticRegression
                classifier=LogisticRegression()
                classifier.fit(X_train,y_train)
                
                #Predicting
                y_pred=classifier.predict(X_test)
                
                #Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm=confusion_matrix(y_test,y_pred)
                
                for x in range(3):
                    for y in range(3):
                        self.total+=cm[x][y]
                        if x==y:
                            self.s+=cm[x][y]
                self.average+=self.s/self.total
            self.average_accuracy=self.average/self.average_number
            #return self.average_accuracy
lr=AverageAccuracy(300)
lr.work()
av=lr.average_accuracy
