from sklearn.base import BaseEstimator
import pandas as pd


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

    def predict(self, X, y=None):
        try:
            getattr(self, "colname")
        except AttributeError:
            raise RuntimeError("You must fit classifier before predicting")
        return [int(x == (self.k - 1)) for x in X[self.colname]]
