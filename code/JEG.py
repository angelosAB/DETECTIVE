import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from sklearn.cluster import KMeans
rpy2.robjects.numpy2ri.activate()
robjects.r.source("C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\methods\JointEmbedding\R\joint_embedding.R")

def JEG(A,K=5,p=20):
    U = robjects.r.multidembed([A[i, :, :] for i in range(A.shape[0])], p)
    Embeddings = list(U)[2]
    model_KM = KMeans(n_clusters=K).fit(Embeddings)
    return(model_KM.labels_)
