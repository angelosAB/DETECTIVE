import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import tensorly as tl
from tensorly import decomposition
class CA_tensor(object):
    def __init__(self,A,X):
        # A multi-layer network of shape (L,n,n)
        # X covariate matrix: n * p
        self.A = A
        self.X = X
        self.XXT = X @ X.T
        self.n = self.A.shape[1]
        self.A_mat = [self.A[i,:,:] for i in range(self.A.shape[0])]
    def Get_clusters(self,a,K,mul=False):
        self.alpha = a
        Temp_A = self.A_mat.copy()
        Temp_A.append(self.alpha * self.XXT)
        Temp_A_mat = np.concatenate(Temp_A,axis=1)
        #U = np.linalg.svd(Temp_A_mat)[0]
        U_1 = np.linalg.svd(Temp_A_mat @ Temp_A_mat.T)[0][:, 0:K]
        U_1 = normalize(U_1)
        if mul!=False:
            Mem_set = []
            for i in range(mul):
                KM = KMeans(n_clusters=K,n_init=50).fit(U_1)
                Mem_set.append(KM.labels_)
                print(i)
            return(Mem_set)
        else:
            KM = KMeans(n_clusters=K,n_init=50).fit(U_1)
            self.Membership = KM.labels_
            return(self.Membership)
    def Get_clusters_tucker(self,a,K):
        self.alpha = a
        A_tensor = np.zeros([self.A.shape[0]+1,self.A.shape[1],self.A.shape[2]])
        A_tensor[0:self.A.shape[0],:,:] =self.A
        A_tensor[self.A.shape[0],:,:] = self.alpha * self.XXT
        Min_L = int(min((K+1)*K/2,self.A.shape[0]+1))
        U_1 = decomposition.tucker(A_tensor,[Min_L,K,K])
        U_1 = normalize(U_1.factors[1])
        KM = KMeans(n_clusters=K, n_init=50).fit(U_1)
        self.Membership = KM.labels_
        return (self.Membership)
    def Get_clusters_tucker_n(self,K):
        Min_L = int(min((K+1)*K/2,self.A.shape[0]))
        U_1 = decomposition.tucker(self.A,[Min_L,K,K])
        U_1 = normalize(U_1.factors[1])
        KM = KMeans(n_clusters=K, n_init=100).fit(U_1)
        self.Membership = KM.labels_
        return (self.Membership)



