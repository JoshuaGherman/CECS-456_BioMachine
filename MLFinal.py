"""
Final Project: Student Behavioral Patterns in Online Learning Platform
Group 13
- Ryan Gieg, 018301580
- 
- 
- 
- 
- 
- 
- 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Ingesting custom csv of data
dataset = pd.read_csv('student_data.csv', delimiter=',')
dataset[np.isnan(dataset)] = 0
# Breaking dataframe into potentially useful subsects
finalGrades = dataset[['StudentID','fg1', 'fg2']]
sessionGrades = dataset[['StudentID', 'sg2', 'sg3', 'sg4', 'sg5', 'sg6']]
sessionAttendance = dataset[['StudentID', 'sa1', 'sa2', 'sa3', 'sa4', 'sa5', 'sa6']]

#print(sessionAttendance.iloc[0])

# List of all student ID nums
stdIDList = []
# List of num of sessions attended by each student
attList = [] # Indices correlate with those of stdIDList
# Nested for loops to get separate lists for Student IDs and student attendance frequency
for i, row in sessionAttendance.iterrows():
    attItr = 0
    stdIDList.append(row['StudentID'])
    for i in range(6):
        if row['sa' + str(i+1)] == 1:
            attItr = attItr+1
    attList.append(attItr)
#print(f'{stdIDList[0]} {attList[0]}')

# For loop to determine the num of students that attended each n num of sessions
attFreqList = [0,0,0,0,0,0]
for i in attList:
    attFreqList[i-1] = attFreqList[i-1]+1
#print(attFreqList)

# Removing StudentID from dataset
dataset.drop('StudentID', axis=1, inplace=True)
# Creating a copy of the dataset to be modified
datasetMod = dataset.copy(deep=True)
##############################################################################################
# Bar chart showing frequency with which students attended class
plt.bar([1,2,3,4,5,6], attFreqList)
plt.title('Student Attendance')
plt.xlabel('Number of Sessions Attended')
plt.ylabel('Number of Students')
plt.show()

# Adding attendance to dataframe
datasetMod['Attendance'] = attList

# Combining fg1 and fg2 into a single column of highest scores per student
bestFinalScore = []
for i, row in dataset.iterrows():
    if row['fg1'] > row['fg2']:
        bestFinalScore.append(row['fg1'])
    else:
        bestFinalScore.append(row['fg2'])
datasetMod['BestFinalScore'] = bestFinalScore

# Creating Correlation Matrix
corrMatrix = dataset.corr()
#print(corrMatrix)
# Visulization of Correlation Matrix
sn.heatmap(corrMatrix, annot=True)
plt.show()

dataSimple = datasetMod[['Attendance', 'BestFinalScore']]
corrMatrix = dataSimple.corr()
#print(corrMatrix)
# Visulization of Correlation Matrix
sn.heatmap(corrMatrix, annot=True)
plt.show()

# Distribution of Final Attempts
# Binwidth set to 10 to show typical 10 percent grading bands
sn.displot(datasetMod, x='BestFinalScore', binwidth=10)
plt.show()

# Distribution of session grades for each session
# Binwidth set to 0.5 to group similar grades with appropriate granularity
sn.displot(dataset, x='sg2', binwidth=0.5)
sn.displot(dataset, x='sg3', binwidth=0.5)
sn.displot(dataset, x='sg4', binwidth=0.5)
sn.displot(dataset, x='sg5', binwidth=0.5)
sn.displot(dataset, x='sg6', binwidth=0.5)
plt.show()

# At least some of the outliers in the dataset are the students that only attended
# the first session

##############################################################################################
# K-means Clustering

# Identifying optimal num of clusters for K-means clustering using:
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Elbow Method
def elbowMethod(data):
    distortions = []
    clusterRange = range(1, 15)
    for i in clusterRange:
        kmeanModel = KMeans(n_clusters=i, n_init=10)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)

    plt.figure()
    plt.plot(clusterRange, distortions, 'bx-')
    plt.xlabel('Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method showing the optimal clusters')
    plt.show()

# Silhouette Method
def silhouetteMethod(data):
    range_n_clusters = list(range(2,15))
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, n_init=10)
        preds = clusterer.fit_predict(data)
        score = silhouette_score(data, preds)
        print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

# Using my simplified dataset of Attendance and BestFinalScores
elbowMethod(dataSimple.values) # Results indicate 2-4 clusters optimal
silhouetteMethod(dataSimple.values) # Results indicate 3 clusters optimal from elbow range
# With these results, lets use 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10)
label = kmeans.fit_predict(dataSimple.values)
# Getting Centroids
centroids = kmeans.cluster_centers_
# Getting unique labels
u_labels = np.unique(label)
# plotting the results
for i in u_labels:
    plt.scatter(dataSimple.values[label == i, 0] , dataSimple.values[label == i, 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 20, color = 'k')
plt.legend()
plt.title('K-means Clustering: Simple')
plt.show()

# Using unaltered dataset
scaler = StandardScaler()
datasetSTD = scaler.fit_transform(dataset)
pca = PCA()
pca.fit(datasetSTD)
pca.explained_variance_ratio_
plt.figure()
plt.plot(range(1,14), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Num of Components')
plt.ylabel('Cumlative Explained Variance')
plt.show()
# 5 PC required to achieve 80% variance capture
pca = PCA(n_components=5)
pca.fit(datasetSTD)
pcaScores = pca.transform(datasetSTD)

elbowMethod(pcaScores) # Results indicate 2-5 clusters optimal
silhouetteMethod(pcaScores) # Results indicate 3 clusters optimal from elbow range
# With these results, lets use 4 clusters
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(pcaScores)
# Getting Centroids
centroids = kmeans.cluster_centers_

datasetPCA = pd.DataFrame(pcaScores)
datasetPCA.rename(columns={0:'PC1',1:'PC2',2:'PC3',3:'PC4',4:'PC5'}, inplace=True)
datasetPCA['Cluster'] = kmeans.labels_
xAxis = datasetPCA['PC1']
yAxis = datasetPCA['PC2']
plt.figure()
sn.scatterplot(x=xAxis, y=yAxis, hue=datasetPCA['Cluster'], palette='deep')
plt.scatter(centroids[:,0] , centroids[:,1] , s = 20, color = 'k')
plt.title('K-means Clustering: Full with PCA')
plt.show()

# Utilizing a focused subset of the data, frequency of student attendance and highest
# final attempt, the k-means graph shows a strong connection between a student's grade 
# and their class attendance

##############################################################################################
#HDBscan Clustering

import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage

# Using my simplified dataset of Attendance and BestFinalScores
clusterer = hdbscan.HDBSCAN()
clusterer.fit(dataSimple)
linkageMatrix = linkage(clusterer._single_linkage_tree)
plt.figure()
dendrogram(linkageMatrix)
plt.title('HDBSCAN Dendrogram: Simple')

# Using unaltered dataset
clusterer.fit(dataset)
linkageMatrix = linkage(clusterer._single_linkage_tree)
plt.figure()
dendrogram(linkageMatrix)
plt.title('HDBSCAN Dendrogram: Full')
plt.show()
