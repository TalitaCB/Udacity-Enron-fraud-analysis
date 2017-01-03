#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score,GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


#funções

def remova_nan(data):

    for nome in data:
            for features in data[nome]:
                if data[nome][features] == "NaN":
                    data[nome][features] = float(0)

    return data

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ["email_address", 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']


def remove_zeros(data):
    lista_nomes = []
    for nome in data:
        soma = 0
        for features in data[nome]:
            if features in financial_features:
                soma = soma + data[nome][features]
        if soma == 0:
            lista_nomes.append(nome)

    for nomes in lista_nomes:
        data.pop(nomes, 0)

    return data



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


#email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
 #'shared_receipt_with_poi']

features_total = poi_label + financial_features + email_features
features_sem_poi = financial_features + email_features
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

#Remove outliers
#remove outlier Total
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)


#Após analises de dados  (Ver Projeto Final Machine Learning.ipynb) acho que seria prudente retirar as features 'email_address", "restricted_stock_deferred", "others","loan_advances","director_fees".

#recriando lista de features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']
email_features = ['from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_total = poi_label + financial_features + email_features



### Task 3: Create new feature(s)


#criando novas features

for data in data_dict:

    if data_dict[data]['from_poi_to_this_person'] != "NaN" and data_dict[data]['from_messages'] != "NaN":
        data_dict[data]['indice_poi_to_this_person'] = float(data_dict[data]['from_poi_to_this_person']) / float(
            data_dict[data]['from_messages'])
    else:
        data_dict[data]['indice_poi_to_this_person'] = "NaN"

    if data_dict[data]['from_this_person_to_poi'] != "NaN" and data_dict[data]['from_messages'] != "NaN":
        data_dict[data]['indice_from_this_person_to_poi'] = float(data_dict[data]['from_this_person_to_poi']) / float(
            data_dict[data]['from_messages'])
    else:
        data_dict[data]['indice_from_this_person_to_poi'] = "NaN"

features_list = features_total + ['from_this_person_to_poi','indice_poi_to_this_person']

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


#criando labels e features após o feature selection
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Durante a analise de outlier verifiquei muitas features com NA´s  Antes de selecionar minhas features vou fazer uma limpeza.
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy="mean")
features = imp.fit_transform(features)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

kbe = SelectKBest()
combined_features = FeatureUnion([("kbe", kbe)])
X_features = combined_features.fit(features_train, labels_train).transform(features_train)
scaler = preprocessing.MinMaxScaler()


#random forest
clf_rf = RandomForestClassifier()
parameters = {'min_samples_split': [2,10,20],
'n_estimators' :[2,10,20],
'criterion': ['gini', 'entropy'],
'min_samples_leaf': [1, 5, 10],
'max_features' : ['auto','log2','sqrt']}
clf_rf = GridSearchCV(clf_rf, parameters)
clf_rf.fit(features_train, labels_train)
pred = clf_rf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "random forest accuracy", accuracy
pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec


##Naive Bayes
clf_nb = GaussianNB()

parameters = {
              'features__kbe__k': [10,11,12,13,14,15]
              }

pipe_nb = Pipeline(steps=[('scaler', scaler),("features", combined_features), ('nb', clf_nb)])

clf_nb = GridSearchCV(estimator = pipe_nb,param_grid=parameters)
clf_nb.fit(features_train, labels_train)
pred = clf_nb.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "NB accuracy", accuracy
pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec

#imprime parametros do algoritimo
print clf_nb.best_params_
peso_features = {}
i = 0
for j in kbe.scores_:
        peso_features[features_list[i]] = j
        i += 1
print peso_features



#Svm
svm = SVC()


parameters = {'svm__kernel': ["linear",'rbf'],
              'svm__C': [100,1000,10000],
              'features__kbe__k': [10,11,12,13,14,15]
              }


pipe_svm = Pipeline(steps=[('scaler', scaler),("features", combined_features), ('svm', svm)])
clf_svm = GridSearchCV(estimator = pipe_svm,param_grid=parameters)
clf_svm.fit(features_train, labels_train)
pred = clf_svm.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "SVM accuracy", accuracy
pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec



print clf_svm.best_params_

#Decision tree
dt = tree.DecisionTreeClassifier(min_samples_split=40)
scaler = preprocessing.MinMaxScaler()

parameters = {'dt__criterion': ['gini', 'entropy'],
              'dt__min_samples_split': [2, 10, 20],
              'features__kbe__k': [5,6,7,8,9,10,11,12,13,14,15]
}

pipe_dt = Pipeline(steps=[("features", combined_features), ('dt', dt)])

clf_dt = GridSearchCV(estimator = pipe_dt,param_grid=parameters)
clf_dt.fit(features_train, labels_train)
pred= clf_dt.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
accuracy = clf_dt.score(features_test, labels_test)
print 'DecisionTree:'
print accuracy
pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "precision",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "recall",rec


#melhor classificador NB
clf = clf_nb.best_estimator_



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)



