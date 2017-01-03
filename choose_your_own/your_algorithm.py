#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary




"""

#kneares_neighbopars
from sklearn import neighbors
#k define a quantidade de vizinhos que sera computados para encontrar a semelhanca
#weight define o peso do vizinho, Se for uniform ele da peso igual a todos os vizinhos, se for distance ele muda a peso de acordo com a distancia
clf = neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance')
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
#accuracy = 0.936

"""

#adaboost
from sklearn.ensemble import AdaBoostClassifier
#The maximum number of estimators at which boosting is terminated
clf = AdaBoostClassifier(n_estimators=13,learning_rate=0.22,random_state=387)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
#accuracy = 0.924

"""
#random forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(min_samples_split=2,n_estimators = 10,criterion = "gini", min_samples_leaf = 1,  min_weight_fraction_leaf = 0.0, max_features = 'auto',  )
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
#accuracy = 0.928
"""""


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
