from sklearn.base import BaseEstimator
import numpy as np


class TopKClassifier(BaseEstimator):
    def __init__(self, k=1):
        self.k = k
        assert (type(self.k) == int and self.k > 0), "k must be a positive integer"

    def fit(self, X, y, colname):
        assert len(X) == len(y), "X and y must be the same length"
        assert type(colname) == str, "Target column name must be string"
        assert colname in X, "Column with {} name does not exists in given dataset".format(colname)
        self.colname = colname
        return self

    def predict(self, X):
        try:
            getattr(self, "colname")
        except AttributeError:
            raise RuntimeError("You must fit classifier before predicting")
        return [int(x == (self.k - 1)) for x in X[self.colname]]

    def predict_proba(self, X):
        try:
            getattr(self, "colname")
        except AttributeError:
            raise RuntimeError("You must fit classifier before predicting")
        return np.array(
            [[0, 1] if x == (self.k - 1) else [1, 0] for x in X[self.colname]]
        )


class ConstantClassifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        assert len(X) == len(y), "X and y must be the same length"
        return self

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
