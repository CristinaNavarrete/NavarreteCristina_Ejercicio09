import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("USArrests.csv", index_col = 0)
headers=[]
for col in df.columns: 
    headers.append(col) 
    
mean=df.mean()
var=df.var()
#normalizar
df_sc=(df-mean)/var
cov=np.cov(df_sc.T)
eig_vals, eig_vecs = np.linalg.eig(cov)
print(eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

Y = df_sc.dot(matrix_w)
Y.info()
print(9)
n=Y.shape[0]
print(n)
 
nombre=Y.index[1]

type(nombre)
fig , ax1 = plt.subplots(figsize=(9,7))

ax1.set_xlim(-0.75,0.75)
ax1.set_ylim(-0.4,0.4)

for i in range(n):
    txt=str(Y.index[i])
    ax1.annotate(txt, (Y[0][i],Y[1][i]))