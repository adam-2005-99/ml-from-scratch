import numpy as np
import pytest
from ml_from_scratch import DecisionTreeClassifier


# ----- Impurity Tests --------- #

def test_gini():
    dt = DecisionTreeClassifier()
    impurity = dt.gini(y=[1,1,7,7])
    assert impurity == 0.5

def test_gini_pure():
    dt = DecisionTreeClassifier()
    impurity = dt.gini(y=[1, 1, 1, 1])
    assert impurity == 0
    


# ----- Split Tests --------- #

def test_best_threshold():
    dt = DecisionTreeClassifier()

    X_col = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 1, 1])

    threshold, score = dt.best_threshold(X_col, y)

    assert threshold == 2.5
    assert score == 0



# ----- Prediction Tests -------- #

def test_prediction():
    dt = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2
    )

    X_train = [
        [1, 1],
        [2, 2],
        [3, 3],
        [5, 5],
        [6, 6],
        [7, 7],
    ]

    y_train = [0, 0, 0, 1, 1, 1]

    dt.fit(X_train, y_train)

    predictions = dt.predict([
        [1.5, 1.5],
        [6.5, 6.5],
    ])

    assert predictions == [0, 1]



# ----- Validation Tests ------ #

@pytest.mark.parametrize("max_depth", [0, -1, 2.5, True])
def test_invalid_max_depth(max_depth):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=max_depth)

@pytest.mark.parametrize("min_samples_split", [0, -1, 1, 2.5, True])
def test_invalid_min_samples(min_samples_split):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=min_samples_split)
        


def test_empty_training_data():
    dt = DecisionTreeClassifier()
    
    with pytest.raises(ValueError):
        dt.fit([],[])


def test_mismatched_training_lengths():
    dt = DecisionTreeClassifier()
    
    X = [[1,2],[3,4]]
    y = [0]
    
    with pytest.raises(ValueError):
        dt.fit(X,y)
        

def test_training_features_shape():
    dt = DecisionTreeClassifier()
    
    with pytest.raises(ValueError):
        dt.fit([1], [0])
        
        
def test_training_labels_shape():
    dt = DecisionTreeClassifier()
    
    with pytest.raises(ValueError):
        dt.fit([[1], [3]], [[1], [0]])
        
        
def test_prediction_feature_count():
    dt = DecisionTreeClassifier()

    X_train = [[1, 2], [2, 3], [5, 6]]
    y_train = [0, 0, 1]

    dt.fit(X_train, y_train)

    with pytest.raises(ValueError):
        dt.predict([[3, 4, 5]])
        

def test_predict_before_fit():
    dt = DecisionTreeClassifier()
    
    with pytest.raises(RuntimeError):
        dt.predict([1,2])
        
        
def test_fit_returns_self():
    dt = DecisionTreeClassifier()

    result = dt.fit(
        [[0, 0], [1, 1]],
        [0, 1],
    )

    assert result is dt
    
    