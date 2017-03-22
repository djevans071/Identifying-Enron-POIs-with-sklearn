#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'bonus',
                 'salary', 'deferred_income', 'feat_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

nonnull_features = {} # dictionary of feature names whose values are the number of people with a non-null value in their respective feature dictionary
nonnull_poi_features = {} # dictionary of feature names whose values are the number of POIs with a non-null value in their respective feature dictionary
feature_types = {} # dictionary of feature types whose values are the datatypes of each feature
poi_list = []
names = []

# loop over names in dataset
for name, features in data_dict.items():
    # create a list of POIs
    if features['poi']:
        poi_list.append(name)
    names.append(name)

    for key, value in features.items():
        if features[key] != 'NaN':
            if key not in nonnull_features: # count up features with non-null values
                nonnull_features[key] = 0
            nonnull_features[key] += 1

            if features['poi']: # count up features of POIs with non-null values
                if key not in nonnull_poi_features:
                    nonnull_poi_features[key] = 0
                nonnull_poi_features[key] += 1

        if key not in feature_types: # for a given feature, make a list of datatypes
            feature_types[key] = []
        feature_types[key].append(type(value))

# populate features_types with sets of unique datatypes
for key, value in feature_types.items():
    feature_types[key] = set(value)

# ----------------------------------------------------------
### Task 2: Remove outliers

# create pandas dataframe of data_dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(data_dict).transpose()
# clean up dataframe by converting `NaN` to np.nan
df = df.applymap(lambda x: np.nan if x == 'NaN' else x)
# remove initial outliers from dataframe and make scatterplot
df = df[~(df.index == 'TOTAL') & ~(df.index.str.startswith('THE '))]
# sns.lmplot('bonus', 'exercised_stock_options',
#            data = df, fit_reg = False)
# plt.tight_layout()
# plt.show()

# detect outliers
df2 = df[['bonus', 'exercised_stock_options', 'poi']]
# print df2[(df2.bonus > 4000000) & (df2.exercised_stock_options > 15000000)]

# remove problematic outliers from data_dict
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# ----------------------------------------------------------
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

 # dictionary of individuals whose values are the ratios of non-null features to total features
feat_ratio = {}

# add feat_ratio to data_dict
for name, feature_dict in data_dict.items():
    # count number of features that aren't null for a given name
    num_nonnull = 0
    # loop over keys and values in features dictionary for a given name
    for key, value in feature_dict.items():
        if value != 'NaN':
            num_nonnull += 1
        elif key == 'feat_ratio':
            num_nonnull += 0

    feature_dict['feat_ratio'] = num_nonnull/21.

my_dataset = data_dict

# first look through all features and use univariate feature_selection
all_features = feature_types.keys() + ['feat_ratio']
# remove email_address deferral_payments, director_fees, loan_advances and restricted_stock_deferred from all_features
to_remove = ['email_address', 'deferral_payments', 'director_fees',
             'loan_advances', 'restricted_stock_deferred']
for feat in to_remove:
    all_features.pop(all_features.index(feat))

# place poi at the front of the list
all_features.insert(0, all_features.pop(all_features.index('poi')))


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features)
labels, features = targetFeatureSplit(data)

# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=6)
selector.fit(features,labels)

scores = selector.scores_
scores /= scores.max()

# make bar plot of features and scores
all_features.pop(0)
# fselect_series = pd.Series(scores, index = all_features)
# fselect_series = fselect_series.sort_values(ascending = False).plot.bar()
# plt.xticks(rotation = 50, ha = 'right')
# plt.ylabel('scores')
# plt.tight_layout()
# plt.show()

# transform the features variable to only contain the 6 highest scoring features
# features = selector.transform(features)
# OR manually select features and re-implement featureFormat and targetFeatureSplit
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# ----------------------------------------------------------

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

clf_dict = {"Random Forest": RandomForestClassifier(n_estimators = 10,
                                min_samples_split=25,
                                min_samples_leaf=1,
                                criterion="entropy"),
            "Decision Tree": DecisionTreeClassifier(max_depth = 7,
                                            max_features = 'sqrt',
                                            min_samples_split = 4),
            'Naive Bayes': GaussianNB(),
            'KNeighbors': KNeighborsClassifier()
            }

for name, clf in clf_dict.items():
    t0 = time()
    clf.fit(features_train, labels_train)
    print 'Fitting time: {} s'.format(round(time() - t0, 4))
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, features, labels, cv = cv)
    print "{} mean accuracy for 15 CVs: {} +/- {}".format(name, round(np.mean(scores), 3), round(np.std(scores), 3))
    print '\n'

# ------------------------------------------------------------

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# GridSearch for KNeighbors classifier
from sklearn.model_selection import GridSearchCV
params = [{'weights': ['uniform'],
          'n_neighbors': [3,4,5,6],
          'leaf_size': [i for i in xrange(10,31)],
          'p': [1,2]},
          {'weights': ['distance'],
            'n_neighbors': [3,4,5,6],
            'leaf_size': [i for i in xrange(10,31)],
            'p': [1,2]}]

kn_clf = GridSearchCV(clf_dict['KNeighbors'], params, cv = 10,
                        scoring = 'accuracy')
kn_clf.fit(features_train, labels_train)
print 'Best Parameters'
print kn_clf.best_estimator_

kn_clf = kn_clf.best_estimator_

# clf0 = clf_dict['KNeighbors']
# clf0.fit(features_train, labels_train)
# print 'Unoptimated KNeighbors (test)', clf0.score(features_test, labels_test)
# print 'Optimized KNeighbors (test)', kn_clf.score(features_test, labels_test)
# print '\n'

# use test_classifier
# nb_clf = clf_dict['Naive Bayes']
# test_classifier(nb_clf, my_dataset, features_list)
test_classifier(clf_dict['KNeighbors'], my_dataset, features_list)
test_classifier(kn_clf, my_dataset, features_list)

clf = kn_clf

# --------------------------------------------------------------

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
