# coding=utf-8
# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import math


# define tree node with following attributes
class node:
    def __init__(self, col=-1, value=None, results=None, t_tree=None, f_tree=None):
        self.col = col  # column index of the data
        self.value = value  # define a judgment-condition,
        # if the number in column matching with this value, we get true, otherwise we get false
        self.results = results  # save the target value
        self.t_tree = t_tree  # when the output of judgment condition is true, tb represents all data in true subtree
        self.f_tree = f_tree  # when the output of judgment condition is false, fb represents all data in false subtree


# The function of computing entropy
def entropy(data):
    values, counts = np.unique(data[:, -1], return_counts=True)  # get unique value and counts of last column
    e = 0
    for i in range(len(values)):
        q = float(counts[i]) / counts.sum()
        e = e - (q * math.log(q, 2))
    return e


# grouping my_data by column, return separated lists and counts of unique value
def divide_set(my_data, column):
    values, counts = np.unique(column, return_counts=True)
    set_list = []
    for v in values:
        s = my_data[column[:] == v]
        set_list.append(s)
    return set_list, values


# The function of building a tree using ID3 algorithm.
# Calculate the information gain of each attribute, choosing the attribute with maximum gain as node
def build_tree(my_data):
    # initialise values for best split
    best_gain = 0.0
    best_judge = None
    best_sets = None

    column_count = my_data.shape[1] - 1  # the number of feature columns
    for col in range(column_count):
        gain = entropy(my_data)
        # split regarding distinct column value
        set_list, values = divide_set(my_data, my_data[:, col])
        # calculate information gain
        for i in range(len(set_list)):
            entropy_c = entropy(np.asarray(set_list[i]))  # calculate entropy of each subset
            p = float(len(set_list[i])) / float(my_data.shape[0])
            gain = gain - p * entropy_c  # information gain

        # find attribute with maximum information gain and save this attribute as tree node
        if gain > best_gain:
            best_gain = gain
            best_judge = (col, values[0])
            best_sets = set_list

    # when we get a best valid information gain, then constructing subtrees
    if best_gain > 0:
        true_subtree = build_tree(best_sets[0])
        false_subtree = build_tree(best_sets[1])
        new_node = node(col=best_judge[0], value=best_judge[1], t_tree=true_subtree, f_tree=false_subtree)
        return new_node
    else:
        return node(results=set_list[0][0, -1])
        # when the information gain is 0, which means we reaching leaf nodes, return the target value in the last column


# function of prediction
# reference: https://zhuanlan.zhihu.com/p/20794583 author:Yaopeng Huang
def predict(test, tree):
    if tree.results is not None:  # only leaf nodes have results value.
        return tree.results
    else:
        v_test = test[tree.col]  # v_test: test value at tree node
        sub_tree = None
        if v_test == tree.value:  # the judgment-condition is true
            sub_tree = tree.t_tree
        else:
            sub_tree = tree.f_tree  # the judgment-condition is false
        return predict(test, sub_tree)  # predict recursively


# print decision tree
# def print_tree(tree, indent=''):
#     if tree.results != None:
#         print str(tree.results)
#     else:
#         print str(tree.col) + ":" + str(tree.value) + "? "
#         print indent + "T->",
#         printtree(tree.t_tree, indent + " ")
#         print indent + "F->",
#         printtree(tree.f_tree, indent + " ")


# ClassifierAgent

# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    # function of generating train set and test set from X and y
    # reference: solution of week2 lab "classify-breast cancer solution.py"
    def cross_val(self, X, y, test_ratio):
        row_number = X.shape[0]
        print row_number
        # calculate the length of test set
        test_number = round(float(row_number) * test_ratio)
        print test_number
        # generate a list to save row index
        all_line = range(0, row_number)
        # random sampling
        test_line = random.sample(all_line, int(test_number))
        # find lines which are not in test_line
        train_line = set(all_line) - set(test_line)
        # separate the data into train set and test set
        X_train = X[list(train_line), :]
        X_test = X[test_line, :]
        y_train = y[list(train_line)]
        y_test = y[test_line]
        return X_train, X_test, y_train, y_test

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of integers 0-3 indicating the action
        # taken in that state.

        # *********************************************
        overall_test_size = 0.1  # percentage of data for the test sample (used in most exercises)
        self.tree_list = []  # create a list inclusing all trees we build

        # Using N-fold cross-validation to split the original dataset to training set and test set
        # build N decision trees using different training data
        N = 10
        for t in range(N):
            # X_train, X_test, y_train, y_test = train_test_split(np.asarray(self.data), np.asarray(self.target), test_size=overall_test_size,
            # random_state=None)
            X_train, X_test, y_train, y_test = self.cross_val(np.asarray(self.data), np.asarray(self.target),
                                                              overall_test_size)
            trans = y_train.reshape(X_train.shape[0], 1)
            self.my_data = np.hstack((X_train, trans))  # combining X_train and target value to one array
            self.pm_tree = build_tree(self.my_data)  # build a decision tree
            self.tree_list.append(self.pm_tree)  # saving the tree to the tree_list, we will get N different trees

        # *********************************************

    # Tidy up when Pacman dies
    def final(self, state):
        print "I'm done!"

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        # *****************************************************
        # predicting with each tree in tree_list, and add the predicted value to predict_list
        predic_list = []
        for tree_i in self.tree_list:
            number = predict(features, tree_i)
            predic_list.append(number)

        # calculate the mean of predicts of N trees as result
        # mode_predic = np.argmax(np.bincount(predic_list))
        avg_predic = np.mean(predic_list)
        p = round(avg_predic)
        d = self.convertNumberToMove(p)
        # *******************************************************
        # Get the actions we can try.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # If the predicted move is not legal, just choose one from the legal move list randomly
        if d not in legal:
            d = random.choice(legal)
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(d, legal)
