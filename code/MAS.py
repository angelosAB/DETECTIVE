from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
def spectral_clustering(A, K):
    gamma, v = np.linalg.eig(A)
    gamma = gamma.real
    v = v.real
    gamma = abs(gamma)
    index = gamma.argsort()
    index = list(index)
    index.reverse()
    index = index[:K]
    v = v[:, index]
    v = normalize(v)
    kmeans = KMeans(n_clusters=K,n_init=20).fit(v)
    return kmeans.labels_


def mean_adj(A,K):
    """
    This is the spectral clustering algorithm on the mean adjacency matrix of the multi-layer network.
    :param A: adjacency tensor
    :param K: number of communities
    :return: community assignment
    """
    tensor_type = A.shape
    n, M = tensor_type[1], tensor_type[0]
    A = A.sum(axis=0)/M
    return spectral_clustering(A, K)




