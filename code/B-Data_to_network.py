import pandas as pd
import numpy as np

DF_Stock = pd.read_csv('C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Stock_processing_features\DF_Stock')
Features = np.load('C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Stock_processing_features\Stock_features.npy')
Network = pd.read_csv('C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Stock_processing_features\Stock_connection_DF')

Num_node = DF_Stock.shape[0]
Num_layer = len(Network['Link'].unique())
Sections = DF_Stock['Sector'].unique().tolist()
Labels = np.array([Sections.index(DF_Stock.iloc[i,3]) for i in range(Num_node)])
Company_names = DF_Stock['Company'].to_list()
Layer_names = Network['Link'].unique().tolist()
A = np.zeros([Num_layer,Num_node,Num_node])

Num_link = Network.shape[0]
for i in range(Num_link):
    Node_1_name = Network['Node_1'].iloc[i]
    Node_2_name = Network['Node_2'].iloc[i]
    Link = Network['Link'].iloc[i]
    Index_1 = Company_names.index(Node_1_name)
    Index_2 = Company_names.index(Node_2_name)
    Layer_index = Layer_names.index(Link)
    A[Layer_index,Index_1,Index_2] = 1
    A[Layer_index,Index_2,Index_1] = 1
np.save('Tensor',A)
np.save('Label',Labels)
#------------------------------------------------
import sys
sys.path.append('methods/')
import CA_tensor as CA
import Hamming_error as He
from sklearn.cluster import KMeans
model = CA.CA_tensor(A,Features)

CASBM_error = []
KM_error = []
MLSBM_error = []
for i in range(100):
    Labels_method = model.Get_clusters_tucker(0.2, 11)
    CASBM_error.append(He.HE_fast(Labels, Labels_method))


    Labels_MLSBM = model.Get_clusters_tucker(0.,11)
    MLSBM_error.append(He.HE_fast(Labels,Labels_MLSBM))

    model_KM = KMeans(n_clusters=11).fit(Features)
    KM_error.append(He.HE_fast(Labels,model_KM.labels_))
    print(i)

import matplotlib.pyplot as plt
plt.boxplot([CASBM_error,KM_error,MLSBM_error],labels=['CASBM','K-means','MSBM'])
plt.grid()



#------------------------Major Community-------------------------------
Sections_num = [DF_Stock['Sector'].tolist().count(i) for i in Sections]
Index_includ = np.array([Sections_num[Sections.index(i)]>30 for i in DF_Stock['Sector'].tolist()])
A_tilde = A[:,Index_includ,:][:,:,Index_includ]
F_tilde = Features[Index_includ,:]
L_tilde = Labels[Index_includ]
L_tilde[L_tilde==5]=3
L_tilde[L_tilde==7]=4
np.save('Tensor',A_tilde)
np.save('Community',L_tilde)
model = CA.CA_tensor(A_tilde,F_tilde)
CASBM_error = []
KM_error = []
MLSBM_error = []

for i in range(100):
    Labels_method = model.Get_clusters_tucker(0.075, 5)
    CASBM_error.append(He.hamming_error([L_tilde, Labels_method]))


    Labels_MLSBM = model.Get_clusters_tucker_n(5)
    MLSBM_error.append(He.hamming_error([L_tilde,Labels_MLSBM]))

    model_KM = KMeans(n_clusters=5).fit(F_tilde)
    KM_error.append(He.hamming_error([L_tilde,model_KM.labels_]))
    print(i)

import matplotlib.pyplot as plt
plt.boxplot([CASBM_error,KM_error,MLSBM_error],labels=['CASBM','K-means','MSBM'])
plt.grid()



#-save dictionary
Selected_labels = np.array(DF_Stock['Sector'])[Index_includ]
dict = {'network':A_tilde.tolist(),'feature':F_tilde.tolist(),'labels':L_tilde.tolist(),'Names':Selected_labels.tolist()}
import json
path = 'C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Data.json'
with open(path, "w") as file:
    json.dump(dict, file, indent=2, ensure_ascii = False)

with open(path, 'r') as file:
    text = file.read()
dict_1 = json.loads(text)












import numpy as np
import matplotlib.pyplot as plt
X = np.arange(1,20,1)
W,B=100,2
(W*B)**X*(X/B+X/(W*B-1))-((W*B)**X-1)/(W*B-1)**2




