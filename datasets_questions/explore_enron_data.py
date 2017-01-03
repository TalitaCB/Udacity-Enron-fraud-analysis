#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import *

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
filename = "../final_project/poi_names.txt"

#numero de pessoas no enron_data
#print len(enron_data)

#numero de features
#print len(enron_data['METTS MARK'])

#numero de pessoas marcadas como suspeitos (POI)

i = 0
for pessoa in enron_data:
    if enron_data[pessoa]["poi"]:
        i = i + 1

#print i



#numero de POI no arquivo pi_names.txt
file_object = open(filename, "r")
nomes = []
for line in file_object:
    if line[:1] == "(":
        nomes.append(line)
#print len(nomes)

#total nvalue of stock belonging to james prentice
#print enron_data['PRENTICE JAMES']["total_stock_value"]

#print enron_data['COLWELL WESLEY']["from_this_person_to_poi"]

#print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

#print enron_data['SKILLING JEFFREY K']["total_payments"] '8.682.716'

#print enron_data['FASTOW ANDREW S']["total_payments"] '2.424.083'
#print enron_data.keys()
#print enron_data['LAY KENNETH L']["total_payments"] 103.559.793

#encontrar a quantidade de salarios  e e-mail em branco

i = 0
j = 0
k = 0
for nome in enron_data:
    if enron_data[nome]["salary"] != "NaN":
       i += 1
    if enron_data[nome]["email_address"] != "NaN":
        j += 1
    if enron_data[nome]["total_payments"] == "NaN":
        k +=  1

#print float(k) / len(enron_data)



feature_list = ["poi", "salary", "bonus"]
data_array = featureFormat(enron_data, feature_list)
label, features = targetFeatureSplit(data_array)

k = 0
count_poi = 0
max_exercised_stock_options = 0
min_exercised_stock_options = 999999
for nome in enron_data:
    if enron_data[nome]["poi"] == True:
        count_poi += 1
        if enron_data[nome]["total_payments"] == "NaN":
            k +=  1
        print enron_data[nome]["exercised_stock_options"]
        if enron_data[nome]["exercised_stock_options"] != "NaN":
            if enron_data[nome]["exercised_stock_options"] > max_exercised_stock_options:
                max_exercised_stock_options = enron_data[nome]["exercised_stock_options"]
                #print  min_exercised_stock_options
            if enron_data[nome]["exercised_stock_options"] < min_exercised_stock_options:
                min_exercised_stock_options = enron_data[nome]["exercised_stock_options"]


#print count_poi
#print max_exercised_stock_options
#print min_exercised_stock_options




