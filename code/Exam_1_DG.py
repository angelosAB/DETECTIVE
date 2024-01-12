import numpy as np
import tensorly as tl

def Generate_Z_Exam1(N,K,sed=1):
    np.random.seed(sed)
    C = np.random.choice([i for i in range(K)], N)
    Z = np.eye(K)[C]
    return(Z)


def Generate_B_Exam1(L,K):
    # The probability of all layers are equal
    B_set = []
    for l in range(L):
        np.random.seed(l)
        B_temp = np.triu(np.random.uniform(0.2, 0.5, [K, K]), k=1)
        B_fin = B_temp + B_temp.T + np.diag(np.random.uniform(0.4, 0.5, K))
        B_set.append(B_fin)
    B_all = np.array(B_set)
    return(B_all)
def Generate_A(B,Z,sed=1):
    BZ = tl.tenalg.mode_dot(B, Z, mode=1)
    P = tl.tenalg.mode_dot(BZ, Z, mode=2)
    A_tensor = np.zeros(P.shape)
    np.random.seed(sed)
    for i in range(P.shape[0]):
        A_temp = np.array(P[i, :, :] - np.random.uniform(0, 1, [P.shape[1], P.shape[2]]) > 0) + 0.0
        A_tensor[i, :, :] = np.triu(A_temp, k=1) + np.triu(A_temp, k=1).T
    return(A_tensor)

def Generate_AandX(B,Z,Rep=100,sigma=1,p=6):
    C = (Z * np.array([i for i in range(Z.shape[1])])).sum(axis=1)
    k = len(np.unique(C))
    M = np.concatenate([np.eye(k), np.zeros([k, p - k])], axis=1)
    A_set,X_set = [],[]
    for rep in range(Rep):
        A_t = Generate_A(B, Z,sed=rep)
        np.random.seed(rep)
        X = Z @ M + np.random.normal(0, sigma, [Z.shape[0], M.shape[1]])
        A_set.append(A_t)
        X_set.append(X)
    return(A_set,X_set,C)

def Gene(NN,LL,KK,RR,s_n,sig=0.75):
    Z_temp = Generate_Z_Exam1(NN,KK)
    B_temp = Generate_B_Exam1(LL,KK)
    A_set,X_set,C = Generate_AandX(B_temp * s_n,Z_temp,RR,sigma=sig)
    return(A_set,X_set,C)