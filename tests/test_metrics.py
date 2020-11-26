from models import metrics
import pandas as pd
import numpy as np


def test_report_accuracy():
    y_true = [[1, 0],
              [1, 0],
              [1, 0],
              [1, 0],
              [2, 0],
              [2, 1],
              [2, 0],
              [3, 1],
              [3, 1],
              [3, 1],
              [3, 0]]
    y_true = pd.DataFrame(y_true, columns=["report_id", "is_rootcause"])
    y_true.set_index("report_id", inplace=True)

    y_pred = [0, 0, 0, 0,
              0, 1, 0,
              1, 1, 1, 0]
    result1 = metrics.report_accuracy(y_true, y_pred)
    assert result1 == 1, "TC1 failed"

    y_pred = [0, 0, 1, 0,
              0, 1, 0,
              1, 1, 1, 0]
    result2 = metrics.report_accuracy(y_true, y_pred)
    assert np.isclose(result2, 2 / 3), "TC2 failed"

    y_pred = [0, 0, 0, 0,
              0, 1, 1,
              1, 1, 1, 0]
    result3 = metrics.report_accuracy(y_true, y_pred)
    assert np.isclose(result3, 2 / 3), "TC3 failed"

    y_pred = [1, 0, 0, 0,
              1, 1, 1,
              0, 1, 1, 0]
    result4 = metrics.report_accuracy(y_true, y_pred)
    assert result4 == 0, "TC4 failed"


def test_most_likely_error_accuracy():
    y_true = [[1, 0],
              [1, 0],
              [1, 0],
              [1, 0],
              [2, 0],
              [2, 1],
              [2, 0],
              [3, 1],
              [3, 1],
              [3, 1],
              [3, 0]]
    y_true = pd.DataFrame(y_true, columns=["report_id", "is_rootcause"])
    y_true.set_index("report_id", inplace=True)

    y_proba = np.array([[1, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.1, 0.9],
                        [0.1, 0.9],
                        [0.1, 0.9],
                        [0.9, 0.1]], np.float64)

    result1 = metrics.most_likely_error_accuracy(y_true, y_proba)
    assert np.isclose(result1, 2 / 3), "TC1 failed"

    y_proba = np.array([[0.9, 0.1],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.5, 0.5],
                        [0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.1, 0.9],
                        [0.1, 0.9],
                        [0.1, 0.9],
                        [0.9, 0.1]], np.float64)

    result2 = metrics.most_likely_error_accuracy(y_true, y_proba)
    assert np.isclose(result2, 2 / 3), "TC2 failed"

    y_true = [[1, 1],
              [1, 0],
              [1, 0],
              [1, 0],
              [2, 0],
              [2, 1],
              [2, 0],
              [3, 1],
              [3, 1],
              [3, 1],
              [3, 0]]
    y_true = pd.DataFrame(y_true, columns=["report_id", "is_rootcause"])
    y_true.set_index("report_id", inplace=True)

    y_proba = np.array([[0.9, 0.1],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.5, 0.5],
                        [0.1, 0.9],
                        [0.2, 0.8],
                        [0.8, 0.2],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0.1, 0.9]], np.float64)

    result3 = metrics.most_likely_error_accuracy(y_true, y_proba)
    assert result3 == 0, "TC3 failed"

    y_proba = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.7, 0.3],
                        [0.1, 0.9],
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.2, 0.8],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0, 1]], np.float64)

    result4 = metrics.most_likely_error_accuracy(y_true, y_proba)
    assert result4 == 0, "TC4 failed"
