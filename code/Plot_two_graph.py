import sys
import numpy as np
import json

import pandas as pd

path = '/Business Data/Final_data/Data.json'
with open(path, 'r') as file:
    text = file.read()
dict_1 = json.loads(text)


#-------------------Load data--------------------------
A = np.array(dict_1.get('network'))
X = np.array(dict_1.get('feature'))
C = np.array(dict_1.get('labels'))
Name = np.array(dict_1.get('Names'))
Layer_names = ['partnership', 'customer', 'investment', 'competitor', 'supplier']
Label_names = ['Industrials', 'Health Care', 'Information Technology','Consumer Discretionary', 'Financials']
for l in range(5):
    for c_1 in np.unique(C):
        for c_2 in np.unique(C):
            print(A[3,C==c_1,:][:,C==c_2].mean(),c_1,c_2,3)


node_to_comm = dict()
for i in range(len(C)):
    node_to_comm[i] = C[i]



sys.path.append('Business Data/')
import networkx as nx
import graph_plot as gp
import matplotlib.pyplot as plt
G = nx.from_numpy_matrix(A[3])
pos = gp.Pos_gene(C)
Color = np.array(['green','yellow','orange','blue','black'])
nx.draw_networkx(G, pos, node_color=Color[C],node_size=15,edge_color='purple',with_labels=False)
plt.scatter([-10],[-10],color='green',label=Label_names[0])
plt.scatter([-10],[-10],color='yellow',label=Label_names[1])
plt.scatter([-10],[-10],color='orange',label=Label_names[2])
plt.scatter([-10],[-10],color='blue',label=Label_names[3])
plt.scatter([-10],[-10],color='black',label=Label_names[4])
plt.legend(prop={'size': 6})
plt.show()
plt.xlim(-0.5,8)
plt.ylim(-0.5,9)
plt.grid()



#------------------------------
import seaborn as sns
Conn_mat = []
Name_1 = []
Name_2 = []
for i in np.unique(C):
    for j in np.unique(C):
        Name_1.append(Label_names[i])
        Name_2.append(Label_names[j])
        Conn_mat.append(A[3][C==i,:][:,C==j].mean())
DF = pd.DataFrame({'Sector_1':Name_1,'Sector_2':Name_2,'Prob':Conn_mat})
flights = DF.pivot('Sector_1','Sector_2','Prob')
ax = sns.heatmap(flights)
ax.tick_params(axis='x', colors='red',labelsize=6,labelrotation=1)
ax.tick_params(axis='y', colors='red',labelsize=6)






