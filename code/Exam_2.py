import numpy as np
"""sys import path"""
import sys
sys.path.insert(0, 'C:\D_Disk\CAMSBM_revision\Methods')
sys.path.insert(0, 'C:\D_Disk\CAMSBM_revision\Simulation\Data_generation')
import JEG, LSM, MAS, CAT
from sklearn.cluster import KMeans
import Hamming_error as He
def Evaluation(a):
    Error_NL = []
    A, X, C_0,k,var = a
    lam = 1. / (A.shape[1] * np.sqrt(A.shape[0]) * max(0.012,A.mean()) * var)
    model = CAT.CA_tensor(A, X)
    C_0 = np.array(C_0, dtype='int')
    Labels_method = model.Get_clusters_tucker(lam, k)
    CASBM_E = He.hamming_error([C_0, Labels_method])
    """
    for kkk in [1*1.2**i for i in range(40)]:
        lam = kkk / (A.shape[1] * np.sqrt(A.shape[0]))
        model = CAT.CA_tensor(A, X)
        C_0 = np.array(C_0, dtype='int')
        Labels_method = model.Get_clusters_tucker(lam, k)
        CASBM_E = He.hamming_error([Labels_method,C_0])
        print(kkk,CASBM_E,1./max(0.012,A.mean())/var)
    """

    # ------------------------------K_means-----------------------------
    model_KM = KMeans(n_clusters=k).fit(X)
    KM_E = He.hamming_error([C_0, model_KM.labels_])
    """MSBM"""
    Labels_method = model.Get_clusters_tucker_n(k)
    MSBM_E = He.hamming_error([C_0, Labels_method])
    """LSM"""
    A_tilde = np.zeros([A.shape[0] + 1, A.shape[1], A.shape[2]])
    A_tilde[0:A.shape[0], :, :] = A
    A_tilde[A.shape[0], :, :] = X @ X.T
    # --------------------------------------------------------------------
    while True:
        try:
            model = LSM.LSM(A_tilde)
            L_LSM = model.Run_n(K=k)
            LSM_E = He.hamming_error([C_0, L_LSM])
        # code with possible error
        except:
            print('error')
            continue
        else:
            # the rest of the code
            break
    """MAS"""
    Label_MAS = MAS.mean_adj(A_tilde, K=k)
    Error_MAS = He.hamming_error([C_0, Label_MAS])
    MAS_E = Error_MAS
    # --------------------------------JEG------------------------------
    Label_JEG = JEG.JEG(A_tilde, K=k, p=int(k*(k+1)/2))
    Error_JEG = He.hamming_error([C_0, Label_JEG])
    Error_NL.append([CASBM_E, MSBM_E, KM_E, MAS_E, LSM_E, Error_JEG])
    # Error Concatenation-----------
    return(Error_NL)



if __name__ == '__main__':
    import multiprocessing
    import Exam_2_DG as E2DG
    for s_n in [0.08]:
        N, L, K, Rep,si = 300,40,4,100,0.75
        A_s, X_s, C = E2DG.Gene(N, L, K, Rep,s_n,si)
        for L in [10,15,20,25,30,35,40]:
            alpha = [[A_s[i][0:L, 0:N, 0:N], X_s[i][0:N], C[0:N],K,si] for i in range(Rep)]
            pool = multiprocessing.Pool(processes=5)
            with multiprocessing.Pool(processes=5) as pool:
                result = pool.map(Evaluation, alpha)
            Name = str(L)+str(s_n)
            np.save('C:\D_Disk\CAMSBM_revision\Simulation\Example_2\Results\Exam_2'+Name,np.array(result))
            pool.close()
            pool.join()