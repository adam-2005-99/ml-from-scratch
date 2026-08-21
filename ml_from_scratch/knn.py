from collections import Counter

class KNNClassifier: 

    def __init__(self, k=3, metric="euclidean"):
        """
        K-Nearest Neighbour classifier.

        Parameters:
            k : int
                Number of neighbours used for classification.
            metric : str
                Distance metric ("euclidean" or "manhattan").
        """
        # Check the value of K and Distance Matrices
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        if metric not in{"euclidean", "manhattan"}:
            raise ValueError(
                "metric must be 'euclidean' or 'manhattan'."
            )
        
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    
    def fit(self, X_train, y_train):
        """
        Store the training data.

        Parameters:
            X_train : training data (each row is a sample, each column is a feature)
            y_train : labels for each sample
        """
        
        # Check the samples and the labels size 
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")
        if self.k > len(X_train):
            raise ValueError("k cannot be greater than the number of training samples.")
        
        self.X_train = X_train
        self.y_train = y_train
        
        return self

    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("KNNClassifier must be fitted before prediction.")
  

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Compute Euclidean distance between two samples.
        """
        if len(point1) != len(point2):
            raise ValueError("Points must have the same number of features")
        total = 0
        for p1, p2 in zip(point1, point2):
            total += (p1 - p2) ** 2
        return total ** 0.5


    @staticmethod
    def manhattan_distance(point1, point2):
        """
        Compute Manhattan distance between two samples.
        """
        if len(point1) != len(point2):
            raise ValueError("Points must have the same number of features")
        total = 0
        for p1, p2 in zip(point1, point2):
            total += abs(p1 - p2)
        return total
    

    def compute_distances(self, x_test):
        """
        Compute the distance from one test sample to all training samples.

        Parameters:
            x_test : test data point

        Returns:
            list of tuples in the form of (distance, label).
        """
        # check if model is fitted
        self._check_is_fitted()
        distances = []
        # Loop through training samples
        for x, y in zip(self.X_train, self.y_train):
            
            # Choose distance metric
            if self.metric == "manhattan":
                dist = self.manhattan_distance(x, x_test)
            else:
                dist = self.euclidean_distance(x, x_test)

            distances.append((dist, y))

        return distances


    @staticmethod
    def get_k_nearest(distances, k):
        """
        Return the labels of the k nearest neighbours.
        """
        # Sort distances in ascending order 
        distances.sort(key=lambda item: item[0])
        k_nearest = distances[:k]
        # get the labels of the points
        labels = [label for (_, label) in k_nearest]
        return labels
    

    @staticmethod
    def majority_vote(labels):
        """
        Return the most common label.
        """
        counter = Counter(labels)
        prediction = counter.most_common(1)[0][0]  # get the most common label
        return prediction
    

    # Predict label of one point
    def predict_one(self,x_test):
        """
        Predict the label for a single sample.
        """
        # Compute distance between test point and trainig points
        distances = self.compute_distances(x_test)

        # Get the k nearst labels to test point
        labels = self.get_k_nearest(distances=distances, k=self.k)

        # Get the most repeted label 
        prediction = self.majority_vote(labels=labels)
        # Return prediction
        return prediction
    


    def predict(self, X_test):
        """
        Predict labels for multiple points.

        Parameters:
        X_test : test data

        Returns:
            predicted labels for all samples
        """
        predictions = []
        # Make prediction for each test point
        for x in X_test:
            pred = self.predict_one(x)
            predictions.append(pred)

        return predictions

