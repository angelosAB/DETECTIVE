import numpy as np

for s_n in [0.08]:
    for L in [10, 15, 20, 25, 30, 35, 40]:
        Name = str(L) + str(s_n)
        A = np.load('C:\D_Disk\CAMSBM_revision\Simulation\Example_2\Results\Exam_2'+Name+'.npy',allow_pickle=False)[:,0,:]
        Mean = np.round(np.array(A).mean(axis=0), decimals=4)
        SD = np.round(np.array(A).std(axis=0) / np.sqrt(100), decimals=4)
        print('(' + str(L) + ',',
              str(Mean[0]) + '&',
              str(Mean[1]) + '&',
              str(Mean[2]) + '&',
              str(Mean[3]) + '&',
              str(Mean[4]) + '&',
              str(Mean[5])+'\\')

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-dark")


for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.1'  # very light grey

L_set = [10, 15, 20, 25, 30, 35, 40]
Line_set = ['CAMSBM','MSBM','Kmeans','MAS','LSE','JEG']
Index = [i for i in range(6)]
ALL = [[] for i in range(6)]
Color = ['black',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41', # matrix green
   'red','yellow']
for L in [10, 15, 20, 25, 30, 35, 40]:
    Name = str(L) + str(s_n)
    A = np.load('C:\D_Disk\CAMSBM_revision\Simulation\Example_2\Results\Exam_2' + Name + '.npy', allow_pickle=False)[:,0, :]
    Mean = np.round(np.array(A).mean(axis=0), decimals=4)
    for i in range(6):
        ALL[i].append(Mean[i])
for i in range(6):
    sns.lineplot(L_set,ALL[i],color=Color[i],linewidth=2.,label=Line_set[i])
    sns.scatterplot(L_set,ALL[i],color=Color[i],s=40.)
plt.grid()
plt.xlabel('Number of Layers')
plt.ylabel('Averaged Hamming Errors')