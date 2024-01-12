import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r.source("C:\D_Disk\CAMSBM_revision\Methods\LSM.r")
class LSM(object):
    def __init__(self,A):
        self.A = A
        self.A_shape = A.shape
        self.A_tilde = np.zeros([self.A_shape[0]+1,self.A_shape[1],self.A_shape[2]])
        self.A_tilde[0:self.A_shape[0],:,:] = A
    def Run_n(self,K):
        psi_hat = robjects.r.GetCluster(self.A, K)
        List1 = np.array(psi_hat)
        return(List1-1)
    def Run_y(self,K):
        psi_hat = robjects.r.GetCluster(self.A_tilde, K)
        List1 = np.array(psi_hat)
        return(List1-1)











