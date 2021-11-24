import matplotlib.pyplot as plt
import pandas as pd

#Import Libraries
# from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# import numpy as np

path='F:\\Data science\\Data projects\\K-Mean\\Data For Unspervised.txt'

data=pd.read_csv(path,header=3,names=['effective porosity ', 'Vshale','Gamma'])

X = data.values

# from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt

kmean = KMeans(n_clusters= 2 ,init='random', #also can be random
                             random_state=33,algorithm= 'auto',verbose=0)

kmean.fit(X)

result = kmean.labels_

print(silhouette_score(X , result))






    
y_kmeans = kmean.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'y')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'g')

plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s = 100, c = 'y')
plt.show()

#Calculating Prediction
# y_pred = kmean.predict(X)
# print('Predicted Value for KMeansModel is : ' , y_pred[:])


# print(y_pred.shape)


