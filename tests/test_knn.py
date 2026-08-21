import pytest
from ml_from_scratch import KNNClassifier



# ------ Distance Tests ------ #

def test_euclidean_distance():
    distance = KNNClassifier.euclidean_distance([0, 0], [3, 4])
    assert distance == 5


def test_manhattan_distance():
    distance = KNNClassifier.manhattan_distance([1, 2], [4, 6])
    assert distance == 7


def test_euclidean_mismatched_dimensions():
    with pytest.raises(ValueError):
        KNNClassifier.euclidean_distance(
            [1, 2, 3],
            [1, 2],
        )


def test_manhattan_mismatched_dimensions():
    with pytest.raises(ValueError):
        KNNClassifier.manhattan_distance(
            [1, 2, 3],
            [1, 2],
        )




# ------ Prediction Tests ------ #

def test_predict_euclidean():
    X_train = [
        [0, 0],
        [1, 1],
        [5, 5],
        [6, 6],
    ]
    y_train = [0, 0, 1, 1]

    model = KNNClassifier(k=3, metric="euclidean")
    model.fit(X_train, y_train)

    predictions = model.predict([
        [0.5, 0.5],
        [5.5, 5.5],
    ])

    assert predictions == [0, 1]


def test_predict_manhattan():
    X_train = [
        [0, 0],
        [1, 1],
        [5, 5],
        [6, 6],
    ]
    y_train = [0, 0, 1, 1]

    model = KNNClassifier(k=3, metric="manhattan")
    model.fit(X_train, y_train)

    predictions = model.predict([
        [0.5, 0.5],
        [5.5, 5.5],
    ])

    assert predictions == [0, 1]




# ------ Validation Tests ------ #

@pytest.mark.parametrize("k", [0, -1, 2.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        KNNClassifier(k=k)


def test_invalid_metric():
    with pytest.raises(ValueError):
        KNNClassifier(metric="cosine")


def test_empty_training_data():
    model = KNNClassifier()

    with pytest.raises(ValueError):
        model.fit([], [])


def test_mismatched_training_lengths():
    model = KNNClassifier()

    X_train = [[1, 2], [3, 4]]
    y_train = [0]

    with pytest.raises(ValueError):
        model.fit(X_train, y_train)


def test_k_greater_than_training_samples():
    model = KNNClassifier(k=3)

    X_train = [[1, 2], [3, 4]]
    y_train = [0, 1]

    with pytest.raises(ValueError):
        model.fit(X_train, y_train)


def test_predict_before_fit():
    model = KNNClassifier()

    with pytest.raises(RuntimeError):
        model.predict([[1, 2]])
        

def test_fit_returns_self():
    model = KNNClassifier(k=1)

    result = model.fit(
        [[0, 0], [1, 1]],
        [0, 1],
    )

    assert result is model