from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import joblib

from .preprocessing import balance_labels, get_data_target

import numpy as np
import pandas as pd



classifiers = {
    "RandomForest": {'model':RandomForestClassifier(), 'params': {
        'model__n_estimators': [100, 150, 200, 250],
        'model__max_features': ['sqrt'],
        'model__min_samples_split': [2,3,4,5,8,10,12]
        }},

    "Logistic" : {'model':LogisticRegression(), 'params': {
        'model__penalty': ["l2"],
        'model__solver': ["lbfgs", "newton-cg", "saga", "sag", "liblinear"],
        'model__max_iter' : [400]
        }},

    "KNN" : {'model':KNeighborsClassifier(), 'params': {
        'model__n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70],
        'model__weights': ["uniform", "distance"],
        'model__metric' : ["euclidean", "manhattan", "minkowski", "chebyshev"]
        }}
}


def _clf_pipeline(name, df, balance=True, n_repeat=1):

    scores = []
    reports = []
    models = []
    confusion_matrices = []

    seeds = np.random.randint(0,1000, n_repeat)

    for seed in seeds:

        if balance:
            balanced_df = balance_labels(df, random_state=seed)
            data, target = get_data_target(balanced_df)

        else:
            data, target = get_data_target(df)

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, stratify=target)

        scaler = MinMaxScaler()

        if name not in classifiers.keys():
            name = "RandomForest"
        classifier = classifiers[name]
        model = classifier['model']
        params = classifier['params']

        pipe = Pipeline([('scaler', scaler),('model', model)])

        search = GridSearchCV(pipe, param_grid=params, cv=10, n_jobs=-1, verbose=4)

        search.fit(X_train, y_train)

        y_proba = search.predict_proba(X_test)
        y_predictions = search.predict(X_test)

        f1 = f1_score(y_test, y_predictions, pos_label='Attrited Customer')

        scores.append(f1)
        reports.append(classification_report(y_test, y_predictions))

        confusion_matrices.append(confusion_matrix(y_test, y_predictions))
        models.append(search)

        joblib.dump(models[-1], f'./assets/{name}_gridsearch.pkl')
        joblib.dump(scores[-1], f'./assets/{name}_score.pkl')
        joblib.dump(confusion_matrices[-1], f'./assets/{name}_confusion_matrix.pkl')
        joblib.dump(reports[-1], f'./assets/{name}_classification_report.pkl')
        joblib.dump(data.columns.to_list(), f'./assets/{name}_model_features.pkl')
        joblib.dump((X_train, X_test, y_train, y_test, y_predictions, y_proba), f'./assets/{name}_train_test_predict_proba.pkl')


