from sklearn.cluster import KMeans
import numpy as np

def cluster(arr,num,iter_no):
    """Finds the cluster centers by running KMeans unsupervised algorithm

    Args:
        arr (numpy array): binary image
        num (int): number of clusters
        iter_no (int): number of iterations

    Returns:
        numpy array: consists of cluster centers
    """

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