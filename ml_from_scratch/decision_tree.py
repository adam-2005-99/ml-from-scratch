import numpy as np

class Node:

    def __init__(self, prediction=None, feature_index=None, 
                 threshold=None, left=None, right=None):
    
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.feature_index is None 


class DecisionTreeClassifier:
    """
    A Decision Tree classifier using Gini impurity.

    The tree works by finding the best feature and threshold to split the data,
    then repeating this process recursively until a stopping condition is met.
    """
    def __init__(self, max_depth=6, min_samples_split=4):
        """
        Set the main tree parameters.

        Parameters:
            max_depth : maximum depth allowed for the tree
            min_samples_split : minimum number of samples needed to split a node
        """
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
            raise ValueError("max_depth should be a positive integer")
        if not isinstance(min_samples_split, int) or isinstance(min_samples_split, bool) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer greater than 1")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = None

    @staticmethod
    def gini(y):
        """
        Compute the Gini impurity of a label array.
        """

        if len(y) == 0:
            return 0
        
        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # return squared sum
        return 1 - np.sum(probabilities ** 2)



    def split_gini(self, X_col, y, threshold):
        """
        Calculate the weighted Gini impurity after splitting one feature column
        using a given threshold.

        Parameters:
            X_col : one column of feature values
            y : labels for the samples
            threshold : value used to split the data

        Returns:
            weighted Gini impurity of the split
        """

        left_child = X_col <= threshold
        right_child = X_col > threshold

        y_left = y[left_child]
        y_right = y[right_child]
        if(len(y_left) == 0 or len(y_right) == 0):
            return 1
        
        total = len(y)
        g_left = (len(y_left)/total) * self.gini(y_left)
        g_right = (len(y_right)/total) * self.gini(y_right)

        return g_left + g_right


    def best_threshold(self, X_col, y):
        """
        Find the best threshold for splitting one feature.

        It checks the midpoints between sorted unique values and returns the one
        that gives the lowest Gini impurity.
        """
        values = np.unique(X_col)
        values = np.sort(values)
        
        thresholds = []
        for i in range(len(values) - 1):
            mid = (values[i] + values[i+1]) / 2
            thresholds.append(mid)

        if len(thresholds) == 0:
            return None, float("inf")

        best_threshold_value = None
        best_score = float("inf")

        for threshold in thresholds:
            score = self.split_gini(X_col, y, threshold)
            if score < best_score:
                best_score = score
                best_threshold_value = threshold

        return best_threshold_value, best_score


    def best_split(self, X, y):
        """
        Find the best feature and threshold that minimises Gini impurity.

        Parameters:
            X : training data (each row is a sample, each column is a feature)
            y : labels for each sample

        Returns:
            best_feature : index of the feature to split on
            best_threshold_value : value used to split the data
            best_score : Gini impurity of the split
        """
       
        best_feature = None
        best_threshold_value = None
        best_score = float("inf")

        n_features = X.shape[1]

        for i in range(n_features):
            threshold, score = self.best_threshold(X[:, i], y)

            if threshold is None:
                continue
            if score < best_score:
                best_feature = i
                best_threshold_value = threshold
                best_score= score
        return best_feature, best_threshold_value, best_score


    @staticmethod
    def majority_class(y):
        """
        Return the most common class label in y.
        """
        labels, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        return labels[idx]

    @staticmethod
    def is_pure(y):
        """
        Return True if all labels in y are the same.
        """
        return len(np.unique(y)) == 1




    def build_tree(self, X, y, depth=0):
        """
        Recursively build a decision tree using Gini impurity.
        """
        # Make prediction
        prediction = self.majority_class(y)
        
        # check if pure or max_depth is exceeded
        if self.is_pure(y) or depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(prediction=prediction) 
        
        # get best split
        feature, threshold, score = self.best_split(X, y)

        # if no split possible
        if feature is None:
            return Node(prediction=prediction)

        # split data
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # call recursively for the left and right childs
        left_child = self.build_tree(
            X[left_mask], 
            y[left_mask], 
            depth+1, 
            )
        
        right_child = self.build_tree(
            X[right_mask], 
            y[right_mask], 
            depth+1, 
            )

        return Node(prediction, feature, threshold, left_child, right_child)


    def fit(self, X, y):
        """
        Train the decision tree.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(X) == 0:
             raise ValueError("Training data cannot be empty.")
        if len(X) != len(y):
             raise ValueError("X and y must contain the same number of samples.")
         
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        
        self.root = self.build_tree(X, y)
        self.n_features = X.shape[1]
        return self
    
    def _check_is_fitted(self):
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before prediction")


    def predict_one(self, node, x_row):
        """
        Predict the class label for a single sample.
        """
        if node.is_leaf():
            return node.prediction

        if x_row[node.feature_index] <= node.threshold:
            return self.predict_one(node.left, x_row)
        else:
            return self.predict_one(node.right, x_row)


    def predict(self, X):
        """
        Predict class labels for multiple samples.

        Parameters:
            X : test data

        Returns:
            predicted labels for all samples
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of features as the training data")
        
        predictions = []
        for row in X:
            predictions.append(self.predict_one(self.root, row))
        return predictions