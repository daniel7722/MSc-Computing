#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np

class Node:
    def __init__(self, split_val=None, column=None, label=None):
        self.left = self.right = None
        self.split_val = split_val
        self.column = column
        self.label = label
        
    def add_child(self, child): 
        if self.left == None:
            self.left = child
        elif self.right == None:
            self.right = child
        else:
            print("no children node free")
            exit()


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.root = None

    
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


    
    def induce_decision_tree(self, x, y):
        # check y count is 1 or node column returns -1
        if (len(np.unique(y)) <= 1 or x.shape[0] == 1):
            leaf_node = Node(label=y[0])
            return leaf_node

        else:
            # find best node
            parent = self.find_best_node(x, y)
            
            # get left and right datasets
            if parent.split_val is None:
                label_set, count = np.unique(y, return_counts=True)
                label = label_set[np.argmax(count)]
                leaf_node = Node(label=label)
                return leaf_node
                 
            child_data = self.split_dataset(x, y, parent)
            
            for i in child_data: 
                child_node = self.induce_decision_tree(i[0], i[1])
                parent.add_child(child_node)
        
            return parent
    

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    
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
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
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