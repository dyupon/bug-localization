from functools import partial

import pandas as pd
import os
import shutil
import argparse
import numpy as np
import nltk
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score
from hyperopt import fmin, hp, tpe, Trials, space_eval
from models.metrics import report_accuracy
from models.metrics import most_likely_error_accuracy
from models.baseline_classifiers import TopKClassifier
from models.utils import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from datetime import datetime

report_accuracy_scorer = make_scorer(report_accuracy)
most_likely_error_scorer = make_scorer(most_likely_error_accuracy)


def f_to_min(hps, X, y, model, scorer="roc_auc", ncv=5, fit_params=None, verbose=None):
    if verbose:
        model = model(**hps, verbose=verbose)
    else:
        model = model(**hps)
    if fit_params:
        cv_res = cross_val_score(model, X, y, cv=StratifiedKFold(ncv), scoring=scorer, fit_params=fit_params, n_jobs=-1)
    else:
        cv_res = cross_val_score(model, X, y, cv=StratifiedKFold(ncv), scoring=scorer, n_jobs=-1)
    return -cv_res.mean()


DIR_OUTPUT = "output/"
if __name__ == '__main__':
    startTime = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="skip error-free report 4")
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

    words = [x.split(".") for x in df['frame'].tolist()]
    words = [item for sublist in words for item in sublist]
    counts = pd.Series(nltk.ngrams(words, 3)).value_counts()
    counts = counts[counts >= 1000]
    vocab = [" ".join(x) for x in list(counts.index)]
    df[["frame"]] = df["frame"].str.replace(".", " ")
    vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(3, 3))
    tokenized = vectorizer.fit_transform(df.frame.tolist())
    df[vectorizer.get_feature_names()] = tokenized.toarray()
    cat_cols = cat_cols + vectorizer.get_feature_names()
    del tokenized
    del vectorizer
    df.set_index("report_id", inplace=True)
    X = df.drop(pk_cols, axis=1)
    y = df[["is_rootcause"]]
    X.drop("frame", axis=1, inplace=True)
    y.groupby("report_id").sum().plot.hist(bins=[0, 1, 2, 3, 4, 5, 10]).figure.savefig(
        DIR_OUTPUT + "/target_distribution.png")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        axis="timestamp",
                                                        test_size=0.25)
    del X
    del y
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
    CatBoost OOB
    """
    print("CatBoost OOB...")
    clf = CatBoostClassifier(custom_metric="F1", eval_metric="F1", random_seed=42, thread_count=4, verbose=False)
    clf.fit(X_train, y_train.values.ravel(), cat_features=cat_cols + encode_cols)
    cf_predict = clf.predict(X_test)
    cf_proba = clf.predict_proba(X_test)
    with open(DIR_OUTPUT + "/results.txt", "a") as cb:
        cb.write("\n \n------------- CatBoost OOB ------------- \n")
        cb.write("ROC-AUC score: {} using {} \n".format(roc_auc_score(y_test, cf_proba[:, 1]), clf.get_all_params()))
        cb.write("F1 score for CatBoostClassifier: {:.3f} \n".format(metrics.f1_score(y_test, cf_predict)))
        cb.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, cf_predict)))
        cb.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, cf_proba)))
        cb.write(np.array2string(metrics.confusion_matrix(y_test, cf_predict)))

    """
    CatBoost hyperopt
    """
    print("CatBoost hyperopt...")
    space4cb = {
        'depth': hp.choice('depth', range(1, 15)),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
        'l2_leaf_reg': hp.choice('l2_leaf_reg', range(1, 10)),
        'border_count': hp.choice('border_count', range(1, 5)),
        'iterations': hp.choice('iterations', [100, 150, 300, 500])
    }
    trials_cb = Trials()
    best_clf = fmin(partial(f_to_min, X=X_train, y=y_train.values.ravel(), model=CatBoostClassifier,
                            fit_params={"cat_features": cat_cols + encode_cols}, verbose=False),
                    space4cb, algo=tpe.suggest, max_evals=100,
                    trials=trials_cb, rstate=np.random.RandomState(42))
    clf = CatBoostClassifier(**space_eval(space4cb, best_clf))
    clf.fit(X_train, y_train.values.ravel(), cat_features=cat_cols + encode_cols)
    cf_proba = clf.predict_proba(X_test)
    clf_val_score = roc_auc_score(y_test, cf_proba[:, 1])
    cf_predict = clf.predict(X_test)
    with open(DIR_OUTPUT + "/results.txt", "a") as cb:
        cb.write("\n \n------------- CatBoost hyperopt ------------- \n")
        cb.write("Cross-val score: {} \n".format(-trials_cb.best_trial['result']['loss']))
        cb.write("ROC-AUC score: {} using {} \n".format(clf_val_score, space_eval(space4cb, best_clf)))
        cb.write("F1 score for RF Classifier: {:.3f} \n".format(metrics.f1_score(y_test, cf_predict)))
        cb.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, cf_predict)))
        cb.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, cf_proba)))
        cb.write(np.array2string(metrics.confusion_matrix(y_test, cf_predict)))
        cb.write("\n Feature importance \n")

    df_onehot_dropfirst = df
    for cat_col in encode_cols:
        if len(df[cat_col].unique()) == 1:
            encoded_col = pd.get_dummies(df[cat_col], prefix=cat_col)
            df_onehot_dropfirst = pd.concat([df_onehot_dropfirst.drop(cat_col, axis=1), encoded_col], axis=1)
        else:
            encoded_col = pd.get_dummies(df[cat_col], drop_first=True, prefix=cat_col)
            df_onehot_dropfirst = pd.concat([df_onehot_dropfirst.drop(cat_col, axis=1), encoded_col], axis=1)
    X = df_onehot_dropfirst.drop(pk_cols, axis=1)
    y = df_onehot_dropfirst[["is_rootcause"]]
    X.drop("frame", axis=1, inplace=True)
    del df_onehot_dropfirst
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        axis="timestamp",
                                                        test_size=0.25)
    del X
    del y
    X_train.drop("timestamp", axis=1, inplace=True)
    X_test.drop("timestamp", axis=1, inplace=True)
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
    print("RF with hyperopt...")
    space4rf = {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
        'n_estimators': hp.choice('n_estimators', range(1, 20)),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'class_weight': hp.choice('class_weight', [None, 'balanced'])
    }

    trials_clf = Trials()
    best_clf = fmin(partial(f_to_min, X=X_train, y=y_train.values.ravel(), model=RandomForestClassifier),
                    space4rf, algo=tpe.suggest, max_evals=100,
                    trials=trials_clf, rstate=np.random.RandomState(42))
    clf = RandomForestClassifier(**space_eval(space4rf, best_clf))
    clf.fit(X_train, y_train.values.ravel())
    rf_proba = clf.predict_proba(X_test)
    clf_val_score = roc_auc_score(y_test, rf_proba[:, 1])
    rf_predict = clf.predict(X_test)
    feature_importance = pd.DataFrame(clf.feature_importances_,
                                      index=X_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    with open(DIR_OUTPUT + "/results.txt", "a") as rff:
        rff.write("\n \n------------- RF GridSearchCV ------------- \n")
        rff.write("Cross-val score: {} \n".format(-trials_clf.best_trial['result']['loss']))
        rff.write("ROC-AUC score: {} using {} \n".format(clf_val_score, space_eval(space4rf, best_clf)))
        rff.write("F1 score for RF Classifier: {:.3f} \n".format(metrics.f1_score(y_test, rf_predict)))
        rff.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, rf_predict)))
        rff.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, rf_proba)))
        rff.write(np.array2string(metrics.confusion_matrix(y_test, rf_predict)))
        rff.write("\n Feature importance \n")
        rff.write(feature_importance.to_string())

    """
    LR 
    """
    print("LR with GridSearchCV...")
    scaler = StandardScaler()
    X_train_std = X_train
    X_test_std = X_test
    X_train_std[num_cols] = scaler.fit_transform(X_train_std[num_cols])
    X_test_std[num_cols] = scaler.transform(X_test_std[num_cols])

    space4lr = {
        'C': hp.loguniform('logit.C', -4.0 * np.log(10.0), 4.0 * np.log(10.0)),
        'class_weight': hp.choice('logit.class_weight', [None, 'balanced'])
    }

    trials_clf = Trials()
    best_clf = fmin(partial(f_to_min, X=X_train_std, y=y_train.values.ravel(), model=LogisticRegression),
                    space4lr, algo=tpe.suggest, max_evals=100,
                    trials=trials_clf, rstate=np.random.RandomState(42))

    clf = LogisticRegression(**space_eval(space4lr, best_clf))
    clf.fit(X_train_std, y_train.values.ravel())
    lr_proba = clf.predict_proba(X_test_std)
    clf_val_score = roc_auc_score(y_test, lr_proba[:, 1])
    lr_predict = clf.predict(X_test_std)
    with open(DIR_OUTPUT + "/results.txt", "a") as lf:
        lf.write("\n \n------------- LR hyperopt ------------- \n")
        lf.write("Cross-val score: {} \n".format(-trials_clf.best_trial['result']['loss']))
        lf.write("ROC-AUC score: {} using {} \n".format(clf_val_score, space_eval(space4lr, best_clf)))
        lf.write("F1 score for LR Classifier: {:.3f} \n".format(metrics.f1_score(y_test, lr_predict)))
        lf.write("Accuracy for reports: {:.3f} \n".format(report_accuracy(y_test, lr_predict)))
        lf.write("Accuracy for most likely rootcauses: {:.3f} \n".format(most_likely_error_accuracy(y_test, lr_proba)))
        lf.write(np.array2string(metrics.confusion_matrix(y_test, lr_predict)))

    print(datetime.now() - startTime)
