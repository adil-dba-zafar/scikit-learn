"""
PU (positive/unlabeled) learning demo on text.

This demo showcases various ways of constructing a text classifier that can
distinguish one of twenty newsgroups from all other groups, without training
it on the labels of messages from the other groups.
"""

# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# Copyright (c) 2011 University of Amsterdam
# License: three-clause BSD

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.semisupervised import OneDNFTransformer
from sklearn.utils import check_random_state

import numpy as np
import sys

try:
    category = sys.argv[1]
except IndexError:
    category = 'sci.space'

print "Fetching/loading 20 newsgroups data set"
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

try:
    y_target = data_train.target_names.index(category)
except ValueError:
    printf >> sys.stderr, "error: category %s not in dataset" % category
    sys.exit(1)

target_ind = np.where(data_train.target == y_target)[0]


## Set up experiment

# Put test samples in training set as unlabeled examples
data_train.data += data_test.data
y_train = np.negative(np.ones(len(data_train.target) + len(data_test.target)))
y_train[np.where(data_train.target == y_target)] = 1

y_test = data_test.target == y_target

# fit vectorizer on all data
vect = CountVectorizer()
print "Vectorizing training set"
X_train = vect.fit_transform(data_train.data).tocsr()
print "Vectorizing test set"
X_test = vect.transform(data_test.data)


## Train, apply and evaluate classifier

print "1-DNF method"
y_t = OneDNFTransformer(thresh=10).fit_transform(X_train, y_train)
labeled = np.where(y_t != -1)[0]
print "Number of negative examples found: %d" % len(np.where(y_t == 0)[0])
print "Fit classifier"
clf = MultinomialNB().fit(X_train[labeled, :], y_t[labeled, :])

y_pred = clf.predict(X_test)
print "precision =  %.3f" % precision_score(y_test, y_pred)
print "recall    =  %.3f" % recall_score(y_test, y_pred)
print "F1        =  %.3f" % f1_score(y_test, y_pred)
