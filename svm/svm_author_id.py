#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from class_vis import prettyPicture, output_image
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

#kernel define como sera a linearidade da linha, mais reta, mais irregular. Ou seja, se o alogoritimo se ajusta mais ou nao
#C The C parameter trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.
#gamma parameter defines how far the influence of a single training example reaches, with low values meaning far and high values meaning close.
clf = SVC(kernel = "linear", C = 10000)

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
#print pred[26]

#contar quantidade previsoes que consideraram ser do cris
print sum(pred)



#plt.plot(features_test, labels_test)

#plt.show()
#plt.savefig("testsvm.png")

#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc

#accuracy 0.9761 com rbf
#accuracy 0.9800 com linear
#########################################################


