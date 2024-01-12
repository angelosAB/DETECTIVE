import sys
sys.path.append('methods/')
sys.path.append('methods/MTCOV/code/')
import CA_tensor as CA
import MTCOV
import json
import numpy as np
#------------------------------MTCOV-------------------------------
Data_path = 'C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Business Data\Final_data\Data.json'
with open(Data_path, 'r') as file:
    text = file.read()
dict_1 = json.loads(text)
A = np.array(dict_1.get('network'))
X = np.array(dict_1.get('feature'))
C = np.array(dict_1.get('labels'))
ID = np.array([i for i in range(A.shape[1])]).tolist()
model = MTCOV.MTCOV(N=A.shape[1],L=5,C=5,Z=102)
A_tilde = np.zeros([A.shape[1],A.shape[2],A.shape[0]])
for i in range(A.shape[0]):
    A_tilde[:,:,i] = A[i,:,:]
model.fit(A,X,'log',ID,batch_size=300)

A_tilde = np.zeros([A.shape[0]+1,A.shape[1],A.shape[2]])
A_tilde[0:5,:,:] = A
A_tilde[5,:,:] = X @ X.T * 0.01
#------------------------------Joint Graph Embedding---------------------------------
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from sklearn.cluster import KMeans
import Hamming_error as He
rpy2.robjects.numpy2ri.activate()
robjects.r.source("methods\\JointEmbedding\\R\\Joint_embedding.R")
U = robjects.r.multidembed([A_tilde[i,:,:] for i in range(5)],5)
Embeddings = list(U)[2]
model_KM = KMeans(n_clusters=5).fit(Embeddings)
print(He.hamming_error([C,model_KM.labels_]))

#-------------------------------------------------
import LSM
model = LSM.LSM(A_tilde)
L_LSM = model.Run_n(K=5)
print(He.hamming_error([C,L_LSM]))
#----------------------------------------------
Proposed_method = CA.CA_tensor(A, X)
Label_Pm = Proposed_method.Get_clusters_tucker(a=0.01, K=5)
Error_CASBM = He.hamming_error([C, Label_Pm])