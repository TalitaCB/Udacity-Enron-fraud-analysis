#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print data_dict
max_exercised_stock_options = 0
min_exercised_stock_options = 999999
max_salary = 0
min_salary = 999999

for nome in data_dict:

    if  data_dict[nome]["exercised_stock_options"] != "NaN":
        if  data_dict[nome]["exercised_stock_options"] > max_exercised_stock_options:
            max_exercised_stock_options =  data_dict[nome]["exercised_stock_options"]
            # print  min_exercised_stock_options
        if  data_dict[nome]["exercised_stock_options"] < min_exercised_stock_options:
            min_exercised_stock_options =  data_dict[nome]["exercised_stock_options"]
    if  data_dict[nome]["salary"] != "NaN":
        if  data_dict[nome]["salary"] > max_salary:
            max_salary =  data_dict[nome]["salary"]
            # print  min_exercised_stock_options
        if  data_dict[nome]["salary"] < min_salary:
            min_salary =  data_dict[nome]["salary"]

print max_salary
print min_salary