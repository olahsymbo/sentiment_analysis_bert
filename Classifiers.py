#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:16:14 2020

@author: o.arigbabu.
"""

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import RandomForestClassifier 

class Classifiers:
    
    def __init__(self, param): 
        self.param = param 
        
    def naiveBayes(self, train_data, train_label):
        gnb = BernoulliNB()  
        return gnb.fit(train_data, train_label)

    def randomForest(self, train_data, train_label):
        clfr = RandomForestClassifier(self.param) 
        return clfr.fit(train_data, train_label)
         
    def predictor(self, clfr, test_data): 
        return clfr.predict(test_data)

    def evaluator(self, y_pred, test_label):
        return accuracy_score(test_label, y_pred) 
  
 
        