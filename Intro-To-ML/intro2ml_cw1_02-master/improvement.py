##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from classification import Node
import random

def accuracy(x, y):
    return np.sum(x==y)*100/len(x)

class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self, num_layers_to_prune=None, min_samples_split=2, min_samples_leaf=1):
        self.is_trained = False
        self.root = None
        self.num_layers_to_prune = num_layers_to_prune
        self.max_depth = 0
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    
    def calculate_entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    
    def calculate_info_gain(self, x, y, x_val, sort_col):

        # Calculate total entropy for overall data
        total_entropy = self.calculate_entropy(y)

        # Calculate entropy for left and right of split
        left_entropy = self.calculate_entropy(y[x[:, sort_col] < x_val])
        right_entropy = self.calculate_entropy(y[x[:, sort_col] >= x_val])

        # Calculate info gain
        info_gain = total_entropy - ((len(y[x[:, sort_col] < x_val])/len(y) * left_entropy) 
                                    + (len(y[x[:, sort_col] >= x_val])/len(y) * right_entropy))
        
        return info_gain

        
    def find_best_node(self, x, y):
        y = y.reshape(-1, 1)

        # To keep track of info gain
        max_gain = value_to_split_on = column_to_split_on = None
                
        # loop through each column 
        for i in range(x.shape[1]):
            # sort by that column
            index_list = x[:, i].argsort()
            x = x[index_list]
            y = y[index_list]

            starting_label = y[0]
            starting_val = x[:, i][0]

            # loop through the column
            for x_val, y_val in zip(x[:, i], y):
                if (y_val != starting_label) and (x_val != starting_val):

                    # calculate information gain
                    info_gain = self.calculate_info_gain(x, y, x_val, i)
                    
                    # update the max information gain
                    if max_gain is None or info_gain > max_gain:
                        max_gain = info_gain
                        value_to_split_on = x_val
                        column_to_split_on = i

                    # Update starting label
                    starting_label = y_val
                    starting_val = x_val
                    
        return Node(value_to_split_on, column_to_split_on)


    def split_dataset(self, x, y, node):
        # Simplified dataset splitting
        left_mask = x[:, node.column] < node.split_val
        right_mask = ~left_mask  # Inverse of left_mask
        return (x[left_mask], y[left_mask]), (x[right_mask], y[right_mask])


        
    def induce_decision_tree(self, x, y, depth=0):
        if len(np.unique(y)) <= 1 or x.shape[0] < self.min_samples_split or depth == self.max_depth:
            label_set, count = np.unique(y, return_counts=True)
            leaf_node = Node(label=label_set[np.argmax(count)])
            return leaf_node
        else:
            parent = self.find_best_node(x, y)
            if parent.split_val is None or x.shape[0] <= self.min_samples_leaf:
                label_set, count = np.unique(y, return_counts=True)
                leaf_node = Node(label=label_set[np.argmax(count)])
                return leaf_node
            
            (x_left, y_left), (x_right, y_right) = self.split_dataset(x, y, parent)
            
            if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                parent.left = self.induce_decision_tree(x_left, y_left, depth + 1)
                parent.right = self.induce_decision_tree(x_right, y_right, depth + 1)
            else:
                label_set, count = np.unique(y, return_counts=True)
                return Node(label=label_set[np.argmax(count)])
            
            return parent



    def fit(self, x, y):

        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        self.max_depth = np.inf
        self.root = self.induce_decision_tree(x, y)
        self.max_depth = self.get_max_depth_from_tree() - self.num_layers_to_prune

        self.root = self.induce_decision_tree(x, y)
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    
    def classify_instance(self, instance, node):

        if node.label != None:
            return node.label

        if instance[node.column] >= node.split_val:
            return self.classify_instance(instance, node.right)
        
        return self.classify_instance(instance, node.left)
        
    
    def predict(self, x):

        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=object)
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for i in range(len(x)):
            label = self.classify_instance(x[i], self.root)
            predictions[i] = label
    
        return predictions

    def get_max_depth_from_tree(self):
        return self.root.max_depth()



def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################


    # TODO: Train new classifier

    prev_accuracy = 0
    prev_std = 100
    accuracies = np.zeros(3, dtype=np.float64)
    best_model = None

    # pruning range
    for i in range(6,9):
        
        # min split sample size
        for j in range(4,7):

            # min leaf sample size
            for k in range(1,3):

                # average accuracy
                for n in range(0,3):
                    
                    tempTree = DecisionTreeClassifier(i,j,k)
                    tempTree.fit(x_train, y_train)

                    temp_prediction = tempTree.predict(x_val)
                    accuracies[n] = accuracy(temp_prediction, y_val)
                
                overall_acc = np.mean(accuracies)
                overall_std = np.std(accuracies)
                
                if overall_acc > prev_accuracy:
                    prev_accuracy = overall_acc
                    best_model = tempTree
                    prev_std = overall_std

                if np.isclose(overall_acc, prev_accuracy, atol=0.5) and overall_std < prev_std:
                    prev_accuracy = overall_acc
                    prev_std = overall_std
                    best_model = tempTree

    # set up an empty (M, ) numpy array to store the predicted labels 
    # feel free to change this if needed
    predictions = np.zeros((x_test.shape[0],), dtype=object)
        
    # TODO: Make predictions on x_test using new classifier        


                
    best_prediction = best_model.predict(x_test)

    for i, label in enumerate(best_prediction):
        predictions[i] = label

    # remember to change this if you rename the variable
    return predictions


