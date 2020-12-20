from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math


def count_cv(estimator, df):
    metrics_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=62)
    for train_index, test_index in kf.split(df.drop(["is_rootcause"], axis=1), df["is_rootcause"]):
        fold_train_df = df.loc[train_index, :]
        fold_test_df = df.loc[test_index, :]
        estimator.fit(fold_train_df.drop(["is_rootcause"], axis=1).values, fold_train_df["is_rootcause"].values)
        pred = estimator.predict(fold_test_df.drop(["is_rootcause"], axis=1).values)
        f1 = metrics.f1_score(fold_test_df["is_rootcause"].values, pred)
        metrics_list.append(f1)
    return np.round(np.mean(metrics_list), 4), np.round(np.std(metrics_list), 4)


def train_test_split(df, y, axis, test_size):
    # df = X.copy()
    df["is_rootcause"] = y
    df.sort_values(by=[axis], inplace=True)
    unique_idxs = list(df.index.unique())
    idxs_num = len(unique_idxs)
    threshold = unique_idxs[math.ceil(idxs_num * (1 - test_size))]
    train = df[df.index <= threshold]
    test = df[df.index > threshold]
    return train.drop(["is_rootcause"], axis=1), test.drop(["is_rootcause"], axis=1), train[["is_rootcause"]], test[
        ["is_rootcause"]]
