import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering

path = './artificial/'
name = "xclara.arff"

databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Plot initial data (2D scatter plot)
print("---------------------------------------")
print("Affichage données initiales            " + str(name))
f0 = datanp[:, 0]  # all elements of the first column
f1 = datanp[:, 1]  # all elements of the second column

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : " + str(name))
plt.show()

##################################################################
# k-Nearest Neighbors analysis for determining eps

k = 5  # Define the number of neighbors
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)

# Calculate the average distance to k nearest neighbors, excluding the point itself
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(distances.shape[0])])

# Sort the distances in ascending order for easier visualization
distancetrie = np.sort(newDistances)

# Plot the sorted distances to help determine eps value
plt.title("Plus proches voisins " + str(k))
plt.plot(distancetrie)
plt.show()

##################################################################
# Run DBSCAN clustering method with the chosen parameters

print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon = 2  # Adjusted based on nearest neighbor distances
min_pts = 5  # Minimum number of points for a cluster
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epsilon= " + str(epsilon) + " MinPts= " + str(min_pts))
plt.show()

##################################################################
# Standardization of data

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:, 0]  # all elements of the first column
f1_scaled = data_scaled[:, 1]  # all elements of the second column

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

##################################################################
# DBSCAN on standardized data

print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")
tps1 = time.time()
epsilon = 0.05  # Adjusted for standardized data
min_pts = 5
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title("Données après clustering DBSCAN (2) - Epsilon= " + str(epsilon) + " MinPts= " + str(min_pts))
plt.show()