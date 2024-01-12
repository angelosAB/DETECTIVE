import sys
sys.path.append('Methods/')
import CAT as CA
import json
import numpy as np
from sklearn.cluster import KMeans
import Hamming_error as He
import LSM
import JEG
import MAS
Data_path = 'C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Business Data\Final_data\Data.json'
with open(Data_path, 'r') as file:
    text = file.read()
dict_1 = json.loads(text)
A = np.array(dict_1.get('network'))
X = np.array(dict_1.get('feature'))
C = np.array(dict_1.get('labels'))
Error = []
for i in range(100):
    # ------------------------------CASBM----------------------------------
    lam = 0.075
    model = CA.CA_tensor(A, X)
    Labels_method = model.Get_clusters_tucker(lam, 5)
    CASBM_E =  He.hamming_error([C, Labels_method])
    # ------------------------------K_means-----------------------------
    model_KM = KMeans(n_clusters=5).fit(X)
    KM_E = He.hamming_error([C, model_KM.labels_])
    #-------------------------------MSBM---------------------------
    Labels_method = model.Get_clusters_tucker_n(5)
    MSBM_E = He.hamming_error([C, Labels_method])
    # ---------------------------------LSM--------------------------------
    A_tilde = np.zeros([A.shape[0] + 1, A.shape[1], A.shape[2]])
    A_tilde[0:5, :, :] = A
    A_tilde[5, :, :] = X @ X.T
    #--------------------------------------------------------------------
    model = LSM.LSM(A_tilde)
    L_LSM = model.Run_n(K=5)
    LSM_E = He.hamming_error([C, L_LSM])
    # -----------------------------------MAS---------------------------------
    Label_MAS = MAS.mean_adj(A_tilde, K=5)
    Error_MAS = He.hamming_error([C, Label_MAS])
    MAS_E = Error_MAS
    # --------------------------------JEG------------------------------
    Label_JEG = JEG.JEG(A_tilde)
    Error_JEG = He.hamming_error([C, Label_JEG])
    Error.append([CASBM_E,KM_E,MSBM_E,MAS_E,LSM_E,Error_JEG])
    print(i)

print(np.mean(Error,axis=0))
print(np.std(Error,axis=0)/np.sqrt(len(Error)))

Mean = np.round(np.mean(Error,axis=0), decimals=4)
SD = np.round(np.std(Error,axis=0)/np.sqrt(len(Error)), decimals=4)
