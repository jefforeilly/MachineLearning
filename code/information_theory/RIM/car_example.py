#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functionalities example with car dataset.

Example usage of functionalities provided in: data_processing and
decision_tree.
Compares sklearn scores and decision_tree on one of UCI datasets.
Shows accuracy measurements, number of rules used in each classificator before
and after post pruning (only decision_tree, sklearn does not provide this
functionality [December 2018])

"""

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from data_processing import convert_to_numerical, get_dataset, split_data
from max_ent import MaxEnt

# READ DATA AND RENAME COLUMNS
df = get_dataset(
    './car.data',
    'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data')
df.columns = [
    'buying', 'maintenance', 'doors', 'people', 'lug_boot', 'safety', 'class'
]

# CONVERT STRING VALUES TO THEIR NUMERICAL COUNTERPARTS (FASTER CALCULATION)
convert_to_numerical(
    df,
    columns=[
        'buying', 'maintenance', 'doors', 'people', 'lug_boot', 'safety',
        'class'
    ],
    inplace=True)

# SPLIT DATASET INTO TRAINING, VALIDATION, TESTING
training, validation, test = split_data(df, inplace=True)

# CREATE CLASSIFIER AND FIT IT TO TRAINING DATA
training_X = training.iloc[:, :-1]
training_y = training.iloc[:, -1]
my_clf = MaxEnt()
my_clf.fit(training_X, training_y)

# SPLIT VALIDATION DATASETS INTO X AND y
validation_X = validation.iloc[:, :-1]
validation_y = validation.iloc[:, -1]

# PRINT METRICS FOR PRUNNED AND UNPRUNNED DECISION TREE
my_predictions = my_clf.predict(validation_X)
print("Validation set accuracy: ", accuracy_score(my_predictions,
                                                  validation_y))
test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
my_predictions_test = my_clf.predict(test_X)
print("Test set accuracy: ", accuracy_score(my_predictions_test, test_y))
