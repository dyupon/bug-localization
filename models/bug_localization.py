import pandas as pd
import os
import shutil
import argparse
import numpy as np
import nltk
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from models.metrics import report_accuracy
from models.metrics import most_likely_error_accuracy
from models.baseline_classifiers import TopKClassifier
from models.utils import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from datetime import datetime

DIR_OUTPUT = "output/"
if __name__ == '__main__':
    startTime = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="skip error-free reports 4")
    parser.add_argument("--skip_reports_without_errors", type=str, default="yes")
    config = parser.parse_args()
    DIR_OUTPUT += config.experiment_name
    if os.path.exists(DIR_OUTPUT):
        shutil.rmtree(DIR_OUTPUT)
    os.makedirs(DIR_OUTPUT)
    print("Uploading data...")
    df = pd.read_csv("data3.csv")
    print("Preprocessing...")
    df.dropna(inplace=True)
    df = df.drop(df[(df.days_since_file_changed > -100500) & (df.days_since_file_changed <= -1)].index)
    df = df.replace({'days_since_file_changed': -100500}, -1)
    if config.skip_reports_without_errors == "yes":
        do_not_contain_errors = df.groupby("report_id").filter(lambda x: x["is_rootcause"].sum() == 0).index
        df = df.drop(do_not_contain_errors)

    num_cols = ["line_number", "distance_to_top", "frame_length", "days_since_file_changed",
                "num_people_changed", "file_length", "method_length", "method_num_of_args", "file_num_lines"]
    cat_cols = ["source"]
    encode_cols = ["language", "exception_type"]
    pk_cols = ["issue_id", "file_name"]

    df_onehot_dropfirst = df.copy()
    for cat_col in encode_cols:
        if len(df[cat_col].unique()) == 1:
            encoded_col = pd.get_dummies(df[cat_col], prefix=cat_col)
            df_onehot_dropfirst = pd.concat([df_onehot_dropfirst.drop(cat_col, axis=1), encoded_col], axis=1)
        else:
            encoded_col = pd.get_dummies(df[cat_col], drop_first=True, prefix=cat_col)
            df_onehot_dropfirst = pd.concat([df_onehot_dropfirst.drop(cat_col, axis=1), encoded_col], axis=1)

    df_onehot_dropfirst.set_index("report_id", inplace=True)

    X = df_onehot_dropfirst.drop(pk_cols, axis=1)
    y = df_onehot_dropfirst[["is_rootcause"]]
    del df_onehot_dropfirst
    del df
    words = [x.split(".") for x in X['frame'].tolist()]
    words = [item for sublist in words for item in sublist]
    counts = pd.Series(nltk.ngrams(words, 3)).value_counts()
    counts = counts[counts >= 1000]
    vocab = [" ".join(x) for x in list(counts.index)]
    X[["frame"]] = X["frame"].str.replace(".", " ")
    vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(3, 3))
    tokenized = vectorizer.fit_transform(X.frame.tolist())
    X[vectorizer.get_feature_names()] = tokenized.toarray()
    cat_cols = cat_cols + vectorizer.get_feature_names()
    del tokenized
    del vectorizer
    X.drop("frame", axis=1, inplace=True)
    y.groupby("report_id").sum().plot.hist(bins=[0, 1, 2, 3, 4, 5, 10]).figure.savefig(
        DIR_OUTPUT + "/target_distribution.png")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        axis="timestamp",
                                                        test_size=0.25)
    X_train.drop("timestamp", axis=1, inplace=True)
    X_test.drop("timestamp", axis=1, inplace=True)
    y_train.groupby("report_id").sum().plot.hist(bins=[0, 1, 2, 3, 4, 5, 10]).figure.savefig(
        DIR_OUTPUT + "/train_target_distribution.png")
    y_test.groupby("report_id").sum().plot.hist(bins=[0, 1, 2, 3, 4, 5, 10]).figure.savefig(
        DIR_OUTPUT + "/test_target_distribution.png")
    with open(DIR_OUTPUT + "/diagnostic.txt", "w") as d:
        d.write("Train: ")
        d.write(str(y_train["is_rootcause"].value_counts()))
        d.write("\n")
        d.write("Test: ")
        d.write(str(y_test["is_rootcause"].value_counts()))

    print("Fit/predict: ")
    print("TopKClassifier...")
    """
    Baseline 
    """
    baseline = TopKClassifier()
    baseline.fit(X_train, y_train, colname="distance_to_top")
    baseline_predict = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)
    with open(DIR_OUTPUT + "/results.txt", "a") as baseline:
        baseline.write("------------- BASELINE ------------- \n")
        baseline.write("F1 score for Top1Classifier: {} \n".format(metrics.f1_score(y_test, baseline_predict)))
        baseline.write("Accuracy for reports: {} \n".format(report_accuracy(y_test, baseline_predict)))
        baseline.write(
            "Accuracy for most likely rootcauses: {:.5f} \n".format(most_likely_error_accuracy(y_test, baseline_proba)))
        baseline.write(np.array2string(metrics.confusion_matrix(y_test, baseline_predict)))

    """
    RF OOB
    """
    print("RF OOB...")
    clf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
    clf.fit(X_train, y_train.values.ravel())
    rf_predict = clf.predict(X_test)
    rf_proba = clf.predict_proba(X_test)
    feature_importance = pd.DataFrame(clf.feature_importances_,
                                      index=X_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
    pred_train = np.argmax(clf.oob_decision_function_, axis=1)
    with open(DIR_OUTPUT + "/results.txt", "a") as rff:
        rff.write("\n \n------------- RF OOB ------------- \n")
        rff.write("OOB score: {} \n".format(metrics.f1_score(y_train, pred_train)))
        rff.write("F1 score for OOB RF Classifier: {:.3f} \n".format(metrics.f1_score(y_test, rf_predict)))
        rff.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, rf_predict)))
        rff.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, rf_proba)))
        rff.write(np.array2string(metrics.confusion_matrix(y_test, rf_predict)))
        rff.write("\n Feature importance \n")
        rff.write(feature_importance.to_string())

    """
    RF tuned
    """
    print("RF with GridSearchCV...")
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [2, 4, 6, 8, 10, 15],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 3, 4],
        'max_depth': [2, 3, 4, 5, 6, 7, 10, 15],
        'criterion': ['gini', 'entropy']
    }
    cv_rf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring="f1")
    cv_rf.fit(X_train, y_train.values.ravel())
    clf = cv_rf.best_estimator_
    clf.fit(X_train, y_train.values.ravel())
    rf_predict = clf.predict(X_test)
    rf_proba = clf.predict_proba(X_test)
    feature_importance = pd.DataFrame(clf.feature_importances_,
                                      index=X_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    with open(DIR_OUTPUT + "/results.txt", "a") as rff:
        rff.write("\n \n------------- RF GridSearchCV ------------- \n")
        rff.write("Best score: {} using {} \n".format(cv_rf.best_score_, cv_rf.best_params_))
        rff.write("Mean of scores in CV: {} \n".format(cv_rf.cv_results_['mean_test_score']))
        rff.write("Std of scores in CV: {} \n".format(cv_rf.cv_results_['std_test_score']))
        rff.write("F1 score for RF Classifier: {:.3f} \n".format(metrics.f1_score(y_test, rf_predict)))
        rff.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, rf_predict)))
        rff.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, rf_proba)))
        rff.write(np.array2string(metrics.confusion_matrix(y_test, rf_predict)))
        rff.write("\n Feature importance \n")
        rff.write(feature_importance.to_string())

    """
    CatBoost
    """
    print("CatBoost CV...")
    param_grid = {'depth': [3, 5, 7, 9, 11],
                  'iterations': [10, 50, 100, 150, 300],
                  'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                  'l2_leaf_reg': [3, 1, 5, 10, 100],
                  'border_count': [32, 5, 10, 20, 50, 100, 200]}

    cb_model = CatBoostClassifier(custom_metric="F1", eval_metric="F1", random_seed=42)
    cv_cb = GridSearchCV(estimator=cb_model, param_grid=param_grid, cv=5)
    cv_cb.fit(X_train, y_train.values.ravel(), cat_features=cat_cols)
    clf = cv_cb.best_estimator_
    clf.fit(X_train, y_train.values.ravel())
    cf_predict = clf.predict(X_test)
    cf_proba = clf.predict_proba(X_test)
    with open(DIR_OUTPUT + "/results.txt", "a") as cb:
        cb.write("\n \n------------- CatBoost GridSearchCV ------------- \n")
        cb.write("Best score: {} using {} \n".format(cv_cb.best_score_, cv_cb.best_params_))
        cb.write("Mean of scores in CV: {} \n".format(cv_cb.cv_results_['mean_test_score']))
        cb.write("Std of scores in CV: {} \n".format(cv_cb.cv_results_['std_test_score']))
        cb.write("F1 score for RF Classifier: {:.3f} \n".format(metrics.f1_score(y_test, cf_predict)))
        cb.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, cf_predict)))
        cb.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, cf_proba)))
        cb.write(np.array2string(metrics.confusion_matrix(y_test, cf_predict)))

    """
    LR 
    """
    print("LR with GridSearchCV...")
    scaler = StandardScaler()
    X_train_std = X_train
    X_test_std = X_test
    X_train_std[num_cols] = scaler.fit_transform(X_train_std[num_cols])
    X_test_std[num_cols] = scaler.transform(X_test_std[num_cols])

    grid = {"C": [0.00000001, 0.0001, 0.001, 0.01, 0.1]}
    lr = LogisticRegression(class_weight="auto")
    logreg_cv = GridSearchCV(lr, grid, cv=5, scoring="f1")
    logreg_cv.fit(X_train_std, y_train.values.ravel())
    lr = logreg_cv.best_estimator_
    lr.fit(X_train_std, y_train.values.ravel())
    lr_predict = lr.predict(X_test_std)
    lr_proba = lr.predict_proba(X_test_std)
    with open(DIR_OUTPUT + "/results.txt", "a") as lf:
        lf.write("\n \n------------- LR GridSearchCV ------------- \n")
        lf.write("Best score: {} using {} \n".format(logreg_cv.best_score_, logreg_cv.best_params_))
        lf.write("Mean of scores in CV: {} \n".format(logreg_cv.cv_results_['mean_test_score']))
        lf.write("Std of scores in CV: {} \n".format(logreg_cv.cv_results_['std_test_score']))
        lf.write("F1 score for LR Classifier: {:.3f} \n".format(metrics.f1_score(y_test, lr_predict)))
        lf.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, lr_predict)))
        lf.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, lr_proba)))
        lf.write(np.array2string(metrics.confusion_matrix(y_test, lr_predict)))

    print(datetime.now() - startTime)