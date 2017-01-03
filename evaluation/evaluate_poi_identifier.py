#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import tree

from sklearn import cross_validation
from sklearn.metrics import accuracy_score


#modelo com overfit
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

pred_over = clf.predict(features)


#modelo correto
features_train, features_test,labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

#print accuracy_score(pred, labels_test)

#mede o total de falso positivo
of_true_positives = [(x,y) for x, y in zip(pred,labels_test) if x == y and x == 1.0]
#print "True positives on the Overfitted model: ", len(of_true_positives)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#print precision_score(pred, labels_test)
#print recall_score(pred, labels_test)

#testando como identificar falso posisitov, falso negativo
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

of_true_positives = [(x,y) for x, y in zip(predictions,true_labels) if x == y and x == 1.0]
print "True positives: ", len(of_true_positives)

of_true_negatives = [(x,y) for x, y in zip(predictions,true_labels) if x == y and x == 0.0]
print "True negatives: ", len(of_true_negatives)

of_false_positives = [(x,y) for x, y in zip(predictions,true_labels) if x != y and x == 1.0]
print "False positives: ", len(of_false_positives)

of_false_negatives = [(x,y) for x, y in zip(predictions,true_labels) if x != y and x == 0.0]
print "False negatives: ", len(of_false_negatives)

print precision_score(predictions, true_labels)