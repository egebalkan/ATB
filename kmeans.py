import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import csv


# =============================================================================
# x = pd.read_csv("merged.csv", usecols =[0], header=None)
# y = pd.read_csv("merged.csv", usecols =[1], header=None)
# z = pd.read_csv("merged.csv", usecols =[2], header=None)
# =============================================================================

# =============================================================================
# X=x.values.tolist()
# Y=y.values.tolist()
# Z=z.values.tolist()
# =============================================================================

k =3
max_iter = 100
threshold = 0.5

centroids = {}

#data_file = "Example.tsv"
#df = pd.read_csv(data_file, sep='\t', header=None)
df = pd.read_csv("merged.csv", header=None)

#No labels will be attached so take just data columns
dropped = df[[0,1,2]] 

X = dropped.values.tolist() 


#Euclidean distance
def distance(feat_one, feat_two):
    squared_distance = 0

    for i in range(len(feat_one)):
        squared_distance += (feat_one[i] - feat_two[i])**2

    return sqrt(squared_distance);


for i in range(k):
    centroids[i]=X[i]
    
centroid_points = []
centroid_points.append([centroids[i] for i in range(k)])

errors=[]
for i in range(k):
    centroids[i] = X[i]

centroid_points = []
centroid_points.append([centroids[i] for i in range(k)])

errors = []

for epoch in range(max_iter):
    classes = {}
    for i in range(k):
        classes[i] = [] #K clusters initialization

    #Clustering the points based on distance function to centroids
    for feature in X:
        distances = [distance(feature, centroids[centroid]) for centroid in centroids]
        classification = distances.index(min(distances))
        classes[classification].append(feature)

    error = 0
    
    #Cost(J) calculation (intra-cluster distances)
    for j in classes.keys():
        for point in classes[j]:
            diff = distance(point, centroids[j])
            error += diff * diff 
    
    errors.append(error)
    
    #Terminate
    if epoch > 0:
        if (errors[epoch - 1] - error) < threshold:
            break

    #Recalculation of centroids
    for classification in classes:
        centroids[classification] = np.average(classes[classification], axis = 0)

    centroid_points.append([centroids[i] for i in range(3)])
    
    
    #plot
    
x = pd.read_csv("merged.csv", usecols =[0], header=None)
y = pd.read_csv("merged.csv", usecols =[1], header=None)
z = pd.read_csv("merged.csv", usecols =[2], header=None)
    
X = np.array(X)
C1 = np.array(classes[0])
C2 = np.array(classes[1])
C3 = np.array(classes[2])

fig = plt.figure(figsize = (12,9))
ax = fig.add_subplot(111, projection = '3d')
plt.ylim(-2000,1000)

ax.scatter(C1[:, 0], C1[:, 2], C1[:,1], c = 'r')
ax.scatter(C2[:, 0], C2[:, 2], C2[:,1], c = 'b')
ax.scatter(C3[:, 0], C3[:, 2], C3[:,1], c = 'g')
ax.scatter(centroids[0][0],centroids[0][2],centroids[0][1], marker = '*', color = 'r')
ax.scatter(centroids[1][0],centroids[1][2],centroids[1][1], marker = '*', color = 'b')
ax.scatter(centroids[2][0],centroids[2][2],centroids[2][1], marker = '*', color = 'g')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# =============================================================================
# ax.scatter(x,z,y)
# =============================================================================

plt.show()

