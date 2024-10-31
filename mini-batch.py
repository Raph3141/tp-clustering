"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="birch-rg2.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne


# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
k=25

tps_total = 0
inertie_tot = 0
for i in range(0,25):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    tps_total += round((tps2 - tps1)*1000)
    inertie_tot += model.inertia_

labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie_tot/25, ", runtime = ", tps_total/25,"ms")
#print("labels", labels)



for j in [10,20,50,100,200,500,1000,2000,5000]:
    tps_total = 0
    inertie_tot = 0
    for i in range(0,25):
        tps1 = time.time()
        model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,batch_size=j)
        model.fit(datanp)
        tps2 = time.time()
        tps_total += round((tps2 - tps1)*1000)
        inertie_tot += model.inertia_

    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_

    print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie_tot/25, ", runtime = ", tps_total/25,"ms", "minibatch = ",j)
    #print("labels", labels)

