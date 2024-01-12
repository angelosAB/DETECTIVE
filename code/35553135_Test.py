import numpy as np
"""sys import path"""
import sys
sys.path.insert(0, 'C:\D_Disk\CAMSBM_revision\Methods')
sys.path.insert(0, 'C:\D_Disk\CAMSBM_revision\Simulation\Data_generation')
import LSM
import Exam_2_DG as E2DG
import Hamming_error as He

N, L, K, Rep, si = 600, 5, 4, 20, 0.75
A_s, X_s, C = E2DG.Gene(N, L, K, Rep, 0.2, si)
for i in range(20):
    model = LSM.LSM(A_s[i])
    L_LSM = model.Run_n(K=4)
    LSM_E = He.hamming_error([np.array(C,dtype=int), L_LSM])
    print(LSM_E)