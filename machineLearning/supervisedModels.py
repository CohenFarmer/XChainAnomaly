import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_xgboost_model(X_train, y_train, sample_weights=None, binary=True):
    objective = 'binary:logistic' if binary else 'multi:softprob'
    model = xgb.XGBClassifier(
        objective=objective,
        max_depth = 12,
        learning_rate = 0.075, 
        n_estimators = 1200,
        subsample = 1,
        reg_lambda = 1,
        reg_alpha = 0,
        min_child_weight = 7,
        gamma = 0,
        colsample_bytree = 1.0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def train_random_forest_model(X_train, y_train, sample_weights=None, binary=True):

    model = RandomForestClassifier(
        n_estimators=900,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='log2',
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def train_logistic_regression_model(X_train, y_train, sample_weights=None, binary=True):

    logreg = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        multi_class='auto',
        random_state=42,
    )
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', logreg)
    ])
    if sample_weights is not None:
        pipe.fit(X_train, y_train, logreg__sample_weight=sample_weights)
    else:
        pipe.fit(X_train, y_train)
    return pipe

def test_model(model, X_test, y_test, binary: bool | None = None, threshold: float = 0.5):
    classes = np.unique(y_test)
    if binary is None:
        binary = len(classes) == 2

    if binary and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            y_pred = (proba[:, 1] >= threshold).astype(int)
        else:
            # Fallback to predict if probabilities are not in expected shape
            raw_pred = model.predict(X_test)
            y_pred = raw_pred if raw_pred.ndim == 1 else raw_pred.argmax(axis=1)
    else:
        raw_pred = model.predict(X_test)
        if isinstance(raw_pred, np.ndarray) and raw_pred.ndim == 2:
            y_pred = raw_pred.argmax(axis=1)
        else:
            y_pred = raw_pred

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    if binary:
        print(f"Threshold: {threshold}")
    
    for cls in classes:
        key = str(cls)
        if key in report:
            cls_metrics = report[key]
            print(f"Class {key} — Precision: {cls_metrics['precision']*100:.2f}% | Recall: {cls_metrics['recall']*100:.2f}% | F1: {cls_metrics['f1-score']*100:.2f}%")

    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"Macro Avg — Precision: {macro['precision']*100:.2f}% | Recall: {macro['recall']*100:.2f}% | F1: {macro['f1-score']*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print(df_cm)
