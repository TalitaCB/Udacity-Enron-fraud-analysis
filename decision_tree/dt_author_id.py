#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#min_samples_split define no nos folhas da arvore qual o numero de percentual minio para ele dividir. opr exemplo se um no folha tem 1 porcento e o min_sample igual 2 sginifica que ele nao pode mais ser dividio por 2 e para ai.
#entropia measure of impurity in a bunch of examples. Quantro mais a entropia mais suja e a o exemplo. Formula: Soma -PI * log2 (PI). PI fracao de carros daquele exemplo
#information_gain = entropy(parent) - [weighted average] entropy (chilren). A arvore de decisao maxima o ganho de informacao
#gini mede a qualidade da divisao, como a entropia no sklearn
#Bias (inferencias) -quando o aprendizado de maquina nao conseguie aprender ou aprende errado

#Imprime quantidade features
print len(features_train[0])

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc

#accuracy de 0,9664%
#########################################################


