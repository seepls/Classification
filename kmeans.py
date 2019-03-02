import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

dataset = pd.read_csv('test_incidents.csv')
X = dataset.iloc[ :,[3,4]].values

#splitting data set into training and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# using elbow method to find optimal number of clusters 
wcss = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i ; init = 'k-means++',random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title ('the elbow method')
plt.xlabel('number of clusters ')
plt.ylabel('type of incident')
plt.show()

# fitting kmeans to dataset 
kmeans = KMeans(n_clusters = 5;init = 'k-means++' , random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0] X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Electric')
plt.scatter(X[y_kmeans == 1,0] X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Fire')
plt.scatter(X[y_kmeans == 2,0] X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Health')
plt.scatter(X[y_kmeans == 3,0] X[y_kmeans == 3, 1], s = 100, c = 'yellow', label = 'Road')
plt.scatter(X[y_kmeans == 4,0] X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Nature')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of incidents')
plt.xlabel('incident kind')
plt.ylabel('number of incidents (1-100)')
plt.legend()
plt.show()
