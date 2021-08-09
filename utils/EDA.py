from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.pipeline import Pipeline
from kneed import KneeLocator
from sklearn.metrics import plot_confusion_matrix
from .preprocessing import balance_labels, get_data_target

import numpy as np
import pandas as pd

def cluster(df):

    kmeans = KMeans(
        init="random",
        n_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    dbscan = DBSCAN(eps=0.8)

    data, target = get_data_target(df)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, stratify=target)

    scaler = MinMaxScaler()

    scaler.fit(data)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    data = scaler.transform(data)

    kmeans.fit(data)
    dbscan.fit(data)

    ari_kmeans = ars(target, kmeans.labels_)
    ari_dbscan = ars(target, dbscan.labels_)

    return ari_kmeans, ari_dbscan