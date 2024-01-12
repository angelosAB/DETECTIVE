import numpy as np

for s_n in [0.05,0.1]:
    for N in [300,400,500]:
        Name = str(N) + str(s_n)
        A = np.load('C:\D_Disk\CAMSBM_revision\Simulation\Example_1\Results\Exam_1'+Name+'.npy',allow_pickle=False)[:,0,:]
        Mean = np.round(np.array(A).mean(axis=0), decimals=4)
        SD = np.round(np.array(A).std(axis=0) / np.sqrt(100), decimals=4)
        #print(A)
        print(str(Mean[0]) + '&',
              str(Mean[1]) + '&',
              str(Mean[2]) + '&',
              str(Mean[3]) + '&',
              str(Mean[4]) + '&',
              str(Mean[5])+'\\')
        print(
            '(' + str(SD[0]) + ')' + '&',
            '(' + str(SD[1]) + ')' + '&',
            '(' + str(SD[2]) + ')' + '&',
            '(' + str(SD[3]) + ')' + '&',
            '(' + str(SD[4]) + ')' + '&',
            '(' + str(SD[5]) + ')'+'\\')