import numpy as np
from tensorly.tenalg import multi_mode_dot as mmd
from sklearn.preprocessing import normalize
import tensorly as tl
#-----------------normal function data generation------------------------
def Data_gene_n(n,K=4,L=4,s_n = 0.5,p=10,sigma=0.4):
    # n: the number of nodes
    # K: the number of communities
    # L: the number of layers
    # s_n: the sparsity factor
    # p: dimension of covariates
    # sigma: variance of covariates
    B_set, P_set = [], []
    A = np.zeros([L,n,n])
    M = np.random.uniform(-2,2,[K,p])
    C = np.random.choice([i for i in range(K)], n)
    Z = np.eye(K)[C]
    X = Z @ M + np.random.normal(0,sigma,[n,p])
    for l in range(L):
        B_temp = np.triu(np.random.uniform(0, 1, [K, K]), k=0)
        B_fin = ((B_temp + B_temp.T)-np.diag(np.diag(B_temp))) * s_n
        B_set.append(B_fin)
        P_temp = Z @ B_set[l] @ Z.T
        P_set.append(P_temp)
        A_temp = np.array(P_temp - np.random.uniform(0,1,[n,n])>0)+0.0
        A[l,:,:] = np.triu(A_temp,k=1) + np.triu(A_temp,k=1).T
    return(A,X,C)

def Symmetric(P):
    A_tensor = np.zeros(P.shape)
    for i in range(P.shape[0]):
        A_temp = np.array(P[i,:,:] - np.random.uniform(0,1,[P.shape[1],P.shape[2]])>0)+0.0
        A_tensor[i,:,:] = np.triu(A_temp,k=1) + np.triu(A_temp,k=1).T
    return(A_tensor)


def Generate_B(L_max,K=4,seed1=1):
    np.random.seed(seed1)
    B_set, P_set = [], []
    for l in range(L_max):
        B_temp = np.triu(np.random.uniform(0, 1, [K, K]), k=0)
        B_fin = (B_temp + B_temp.T - np.diag(np.diag(B_temp)))
        B_set.append(B_fin)
    B_all = np.array(B_set)
    return(B_all)



def Data_gene_Exam_1(N_max,L_max,Rep,k = 4,s_n = 0.5,p=10,sigma=0.4):
    # n: the number of nodes
    # K: the number of communities
    # L: the number of layers
    # s_n: the sparsity factor
    # p: dimension of covariates
    # sigma: variance of covariates
    A_set, X_set, C_set = [], [], []
    for rep in range(Rep):
        B_all = Generate_B(L_max,K=k,seed1=rep) * s_n
        np.random.seed(rep)
        M = np.concatenate([2 * np.eye(k), np.zeros([k, p - k])], axis=1)
        C = np.random.choice([i for i in range(k)], N_max)
        C_set.append(C)
        Z = np.eye(k)[C]
        X = Z @ M + np.random.normal(0,sigma,[N_max,p])
        X_set.append(X)
        BZ = tl.tenalg.mode_dot(B_all,Z,mode=1)
        BZZ = tl.tenalg.mode_dot(BZ,Z,mode=2)
        A_temp = Symmetric(BZZ)
        A_set.append(A_temp)
    return(A_set,X_set,C_set)




def Generate_B_Exam2(L_max,K=4,seed1=1):
    # The probability of all layers are equal
    np.random.seed(seed1)
    B_temp = np.triu(np.random.uniform(0, 1, [K, K]), k=0)
    B_fin = (B_temp + B_temp.T - np.diag(np.diag(B_temp)))
    B_set, P_set = [], []
    for l in range(L_max):
        B_set.append(B_fin)
    B_all = np.array(B_set)
    return(B_all)


def Data_gene_Exam_2(L,sigma=1,N = 300,p=6,k=4,s_n=0.04,sed=1):
    X_set = []
    A_set = []
    np.random.seed(sed)
    C = np.random.choice([i for i in range(k)], N)
    Z = np.eye(k)[C]
    for rep in range(100):
        M = np.concatenate([2 * np.eye(k), np.zeros([k, p - k])], axis=1)
        X = Z @ M + np.random.normal(0, sigma, [Z.shape[0], M.shape[1]])
        X_set.append(X)
        B = Generate_B_Exam2(L,seed1=rep)
        B_1 = B * s_n
        BZ = tl.tenalg.mode_dot(B_1, Z, mode=1)
        BZZ = tl.tenalg.mode_dot(BZ, Z, mode=2)
        A_temp = Symmetric(BZZ)
        A_set.append(A_temp)
    return(A_set,X_set,C)
