import numpy as np
def Conductance(A,C,s):
    Shape = A.shape
    n = Shape[1]
    Output = []
    for l in range(Shape[0]):
        A_temp = A[l,:,:]
        S = 0.
        for k in np.unique(C):
            A_temp_k = A_temp[C == k, :][:, C == k]
            O_kk = A_temp_k.sum()
            A_temp_k_out = A_temp[C == k, :][:, C != k]
            print(A_temp_k.sum())
            n_k = sum(C == l)
            if n_k==1 or n_k==0:
                S += 0.
            else:
                S += A_temp_k_out.sum()*2/(A_temp_k.sum()+A_temp_k_out.sum()*2) * n_k/n
        Output.append(S)
    if s==1:
        return(sum(Output))
    else:
        return(np.array(Output))

def Internal_density(A,C,s):
    # A can be tensor (L,n,n)
    # C community assignments
    Shape = A.shape
    n = Shape[1]
    Output = []
    for l in range(Shape[0]):
        A_temp = A[l,:,:]
        S = 0.
        for k in np.unique(C):
            A_temp_k = A_temp[C == k, :][:, C == k]
            O_kk = A_temp_k.sum()
            n_k = sum(C == k)
            if n_k==1 or n_k==0:
                S += 0.
            else:
                S += O_kk/(n_k*(n_k-1)) * n_k/n
        Output.append(S)
    if s==1:
        return(sum(Output))
    else:
        return(np.array(Output))









def Modularity(A,C):
    # A can be tensor (L,n,n) or matrix
    # C community assignments
    Shape = A.shape
    n = Shape[1]
    if len(Shape)==3:
        Output = []
        for l in range(Shape[0]):
            A_temp = A[l,:,:]
            L_l = A_temp.sum()
            S = 0.
            for k in np.unique(C):
                A_temp_k = A_temp[C==k,:][:,C==k]
                O_kk = A_temp_k.sum()
                n_k = sum(C==l)
                S += (O_kk - (n_k/n)**2*L_l)
            Output.append(S/L_l)
