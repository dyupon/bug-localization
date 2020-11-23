def report_accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Ground truth and prediction vectors must be the same length"
    result = y_true.copy()
    result["predicted"] = y_pred
    result = list(result["is_rootcause"].eq(result["predicted"]).astype(bool).groupby("report_id").all().astype(int))
    return result.count(1) / len(result)


def most_likely_error_accuracy(y_true, y_proba):
    assert len(y_true) == len(y_proba), "Ground truth and prediction vectors must be the same length"
    result = y_true.copy()
    result["predicted"] = y_proba[:, 1]
    result = result.groupby("report_id").max()
    return sum(result["is_rootcause"]) / len(result)
