import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
dataset = pd.read_csv("Algorithms/Mall_Customers.csv")

X= dataset.iloc[:,[3,4]].values
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

hc = AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)
print(y_hc)

plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=100,c='yellow',label='Cluster 4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=100,c='orange',label='Cluster 5')
plt.title('Cluster of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.show()