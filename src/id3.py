#!/usr/bin/env python

# Example of execution: id3.py ../data/train.dat ../data/test.dat

from decTree import DecTree
import math
from optparse import OptionParser


# Reads corpus and creates the appropriate data structures, you do not have to change anything in this method
def read_corpus(file_name):
    with open(file_name, 'r') as f:
        # first line contains the list of attributes
        attributes = dict()
        ind = 0
        for attribute in f.readline().strip().split("\t"):
            attributes[attribute] = {'ind': int(ind)}
            ind += 1

        # the rest of the file contains the instances
        instances = []
        ind = 0
        for inst in f.readlines():
            inst = inst.strip()

            elems = inst.split("\t")

            if len(elems) < 3: continue

            instances.append({'values': [int(elem) for elem in elems[0:-1]],
                              'class': int(elems[-1]),
                              'index': int(ind),
                              })
            ind += 1

    return attributes, instances


def all_same_label(instances):
    """ Helper method to check and see if a batch of
    instances has the same class label"""
    label = instances[0]['class']
    for inst in instances[1:]:
        if inst['class'] != label:
            return False
    return True

"""
-- Store decision tree in huge nest of nested dictionaries
"""

def get_majority_label(instances):
    """ Helper method to grab the majority binary label for any given
    set of instances"""
    zero = [instance for instance in instances if instance['class'] == 0]
    one = [instance for instance in instances if instance['class'] == 1]
    if len(zero) > len(one):
        return 0
    else:
        return 1

# A method to calculate the maximum info gain of all remaining attributes
# and return the best attribute to split on

def max_info_gain(instances, attributes):
    winner_attr, winner_left, winner_right = attributes, [], []
    i_g = -2

    for attribute in attributes:

        candidate = attributes[attribute]['ind']

        # Initializing some lists to store the left-right split for the attributes
        # with highest information gain

        def info_gain(candidate, instances):  # Function to calculate the entropy and information gain of any given attribute on its own
            # Entropy math! #
            # Assessing the distribution of binary values
            left = [instance for instance in instances if instance['values'][candidate] == 0]
            right = [instance for instance in instances if instance['values'][candidate] == 1]

            l1 = [instance for instance in left if instance['class'] == 0]
            l2 = [instance for instance in left if instance['class'] == 1]
            r1 = [instance for instance in right if instance['class'] == 0]
            r2 = [instance for instance in right if instance['class'] == 1]

            if len(instances) == 0:
                coef_l, coef_r = 0, 0
            else:
                coef_l = len(left) / len(instances)
                coef_r = len(right) / len(instances)

            if len(left) == 0:
                coef_l1, coef_l2 = 0,0
            else:
                coef_l1, coef_l2 = len(l1) / len(left), len(l2) / len(left)

            if len(right) == 0:
                coef_r1, coef_r2 = 0,0
            else:
                coef_r1, coef_r2 = len(r1) / len(right), len(r2) / len(right)

            if coef_l == 0:
                h_attr_1 = 0
            else:
                h_attr_1 = (coef_l) * math.log2(coef_l)

            if coef_r == 0:
                h_attr_2 = 0
            else:
                h_attr_2 = (coef_r) * math.log2(coef_r)

            h_attr = -(h_attr_1 + h_attr_2)

            ## FIX ME ## Need to be more specific with which of these variables is zero and only control for that
            try:
                h_l = -((coef_l1) * math.log2(coef_l1) + (coef_l2) * math.log2(coef_l2))

            except ValueError:
                h_l = 0

            try:
                h_r = -((coef_r1) * math.log2(coef_r1) + (coef_r2) * math.log2(coef_r2))

            except ValueError:
                h_r = 0

            gain = h_attr - ((h_l * coef_l) + (h_r * coef_r))
            return gain, left, right

        temp = info_gain(candidate, instances)[0]

        if temp > i_g:
            i_g = temp
            winner_attr = attribute
            winner_left = info_gain(candidate, instances)[1]
            winner_right = info_gain(candidate, instances)[2]

    return winner_attr, winner_left, winner_right


def build_tree(attributes, instances, majority):
    ######################################
    # Method to build a tree and predict #
    ######################################

    maj_label = get_majority_label(instances)

    ### Creating base conditions ###
    # If this node is a leaf:
    if len(instances) == 0:
        return DecTree(majority)

    # If we run out of attributes
    if len(attributes) == 0:
        return DecTree(maj_label)

    # If the instances are all the same class (i.e. there is no information gain)
    if all_same_label(instances):
        return DecTree(instances[0]['class'])

    # Passing the attribute with the most information gain back out through the function
    # And letting our tree know what that is
    winner = max_info_gain(instances, attributes)[0]

    # Telling our tree what the left and right branch splits are
    left, right = max_info_gain(instances, attributes)[1], max_info_gain(instances, attributes)[2]

    # Creating copies of the original attributes dictionary
    # And removing the attribute we've already split on from those copies
    attributes_l = dict(attributes)
    attributes_r = dict(attributes)

    del attributes_l[winner]
    del attributes_r[winner]

    # Defining the left and right branches of our newest split of the decision tree
    left_tree = build_tree(attributes_l, left, majority)
    right_tree = build_tree(attributes_r, right, majority)

    return DecTree(winner, left_tree, right_tree)


def predict_class(tree, inst, attributes):
    """ Here is where we will traverse the tree, also recursively,
    and predict a 0 or 1 class for each instance """

    try:
        key = attributes[tree.txt]['ind']  # Accessing the numerical value of the attribute
    except KeyError:
        key = tree.txt                     # But not if we're at a leaf and making a prediction instead of
                                           # accessing an attribute
    if tree.is_leaf():
        return tree.txt
    elif inst['values'][key] == 0:          # Going to the left
        return predict_class(tree.l, inst, attributes)
    else:                                   # Going to the right
        return predict_class(tree.r, inst, attributes)


def calculate_accuracy(tree, attributes, instances):
    """ Short method to calculate number of correct predictions in % format"""

    # Setting counters
    accurate = 0
    counter = 0

    # Looping over instances and comparing
    for instance in instances:
        predicted = int(predict_class(tree, instance, attributes))
        actual = instance['class']
        counter += 1

        if predicted == actual:
            accurate += 1

    return (accurate/len(instances))*100

if __name__ == '__main__':
    usage = "usage: %prog [options] TRAINING_FILE TEST_FILE"

    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--debug", action='store_true',
                      help="Turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Incorrect number of arguments")

    file_tr = args[0]
    file_te = args[1]

    # training instances
    attr_tr, instances_tr = read_corpus(file_tr)
    majority = get_majority_label(instances_tr)

    # DO NOT use file_te to build the tree
    tree = build_tree(attr_tr, instances_tr, majority)
    print(tree)
    print

    # test instances
    attr_te, instances_te = read_corpus(file_te)

    # Accuracy with training is rather useless, but I do want you to see that it is only slightly higher than
    #   the accuracy in test
    accuracy_tr = calculate_accuracy(tree, attr_tr, instances_tr)
    print(f"Accuracy on training set ({len(instances_tr)} instances): {accuracy_tr:.2f}%")
    accuracy_te = calculate_accuracy(tree, attr_te, instances_te)
    print(f"Accuracy on test set     ({len(instances_te)} instances): {accuracy_te:.2f}%")
