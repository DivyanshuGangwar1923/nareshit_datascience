# K-means clustering

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset=pd.read_csv(r"C:\Users\divya\Downloads\Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)    
