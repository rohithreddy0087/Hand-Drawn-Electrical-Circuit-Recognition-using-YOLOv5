# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:48:33 2020

@author: Dell
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def cluster(arr,num,iter_no):

    # A list holds the SSE values for each k
    # feat = np.array(arr)
    # feat = np.squeeze(feat)

    kmeans = KMeans(
        init="random",
        n_clusters=num,
        n_init=10,
        max_iter=iter_no,
        random_state=42
    )
    
    model = kmeans.fit(arr)
    arr = model.cluster_centers_
    
    return arr