"""
Final Project: Student Behavioral Patterns in Online Learning Platform
Group 13
- Ryan Gieg, 018301580
- Jason Jitsiripol, 027462849
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
##############################################################################################
'''
# Bar chart showing frequency with which students attended class
plt.bar([1,2,3,4,5,6], attFreqList)
plt.title('Student Attendance')
plt.xlabel('Number of Sessions Attended')
plt.ylabel('Number of Students')
plt.show()
'''

# Adding attendance to dataframe
dataset['Attendance'] = attList

# Combining fg1 and fg2 into a single column of highest scores per student
bestFinalScore = []
for i, row in dataset.iterrows():
    if row['fg1'] > row['fg2']:
        bestFinalScore.append(row['fg1'])
    else:
        bestFinalScore.append(row['fg2'])
dataset['BestFinalScore'] = bestFinalScore

#setting up x and y for more EDA analysis
X = dataset.iloc[:, :-1].values
y = bestFinalScore

#boxplot of grades
plt.boxplot(y) #shows the score of the students' grades as a boxplot.
plt.show()

#Histogram of the number of grades with a specific score.
plt.hist(y)
plt.show()

# Creating Correlation Matrix
corrMatrix = dataset.corr()
#print(corrMatrix)
# Visulization of Correlation Matrix
sn.heatmap(corrMatrix, annot=True)
plt.show()

dataRel = dataset[['Attendance', 'BestFinalScore']]
corrMatrix = dataRel.corr()
#print(corrMatrix)
# Visulization of Correlation Matrix
sn.heatmap(corrMatrix, annot=True)
plt.show()

'''
# Distribution of 1st Final Attempts
# Binwidth set to 10 to show typical 10 percent grading bands
sn.displot(dataset, x='fg1', binwidth=10)
# Distribution of 2nd Final Attempts
# Binwidth set to 10 to show typical 10 percent grading bands
sn.displot(dataset, x='fg2', binwidth=10)
plt.show()

# Distribution of session grades for each session
# Binwidth set to 0.5 to group similar grades with appropriate granularity
sn.displot(dataset, x='sg2', binwidth=0.5)
sn.displot(dataset, x='sg3', binwidth=0.5)
sn.displot(dataset, x='sg4', binwidth=0.5)
sn.displot(dataset, x='sg5', binwidth=0.5)
sn.displot(dataset, x='sg6', binwidth=0.5)
plt.show()
'''

# Identifying optimal num of clusters for K-means clustering using:
# Elbow Method
from sklearn.cluster import KMeans, dbscan
dataVals = dataRel.values
#dataVals = dataset.values
#dataVals = data
#dataVals = data2

distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init=10)
    kmeanModel.fit(dataVals)
    distortions.append(kmeanModel.inertia_)

plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# The Elbow method indicates that 3 or 4 is the optimal num of clusters

#Alternative Elbow Method
from sklearn import datasets
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,10))
 
visualizer.fit(X)
visualizer.show()
#This Elbow method indicates 4 is optimal, so we need to check both 3 and 4 since 3 is optimal in the elbow method above.

# Silhouette Method
from sklearn.metrics import silhouette_score

silhouettesVal = [] #used to make the graph at the end
range_n_clusters = list(range(2,10))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, n_init=10)
    preds = clusterer.fit_predict(dataVals)
    centers = clusterer.cluster_centers_

    score = silhouette_score(dataVals, preds)
    silhouettesVal.append(score)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
# The Silhouette methods indicates that 2 is the optimal num of clusters

plt.plot(range_n_clusters, silhouettesVal,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()

#Another visual of silhouette method
from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(3, 2, figsize=(15,8))
for i in [2, 3, 4, 5, 6, 7,]:
    #Create KMeans instance for different number of clusters

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
   
    # Create SilhouetteVisualizer instance with KMeans instance and Fit the visualizer

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)
#2 clusters is shown to be the best here since the other ones are way off.

#Clustering attempts
# With these results lets use 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10)
label = kmeans.fit_predict(dataVals)
# Getting Centroids
centroids = kmeans.cluster_centers_
# Getting unique labels
u_labels = np.unique(label)
# plotting the results
for i in u_labels:
    plt.scatter(dataVals[label == i, 0] , dataVals[label == i, 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 20, color = 'k')
plt.legend()
plt.show()
# Utilizing a focused subset of the data, frequency of student attendance and highest
# final attempt, the k-means graph shows a strong connection between a student's grade 
# and their class attendance

# With these results lets use 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10)
label = kmeans.fit_predict(dataVals)
# Getting Centroids
centroids = kmeans.cluster_centers_
# Getting unique labels
u_labels = np.unique(label)
# plotting the results
for i in u_labels:
    plt.scatter(dataVals[label == i, 0] , dataVals[label == i, 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 20, color = 'k')
plt.legend()
plt.show()
# Utilizing a focused subset of the data, frequency of student attendance and highest
# final attempt, the k-means graph shows a strong connection between a student's grade 
# and their class attendance

# With these results lets use 2 clusters
kmeans = KMeans(n_clusters=2, n_init=10)
label = kmeans.fit_predict(dataVals)
# Getting Centroids
centroids = kmeans.cluster_centers_
# Getting unique labels
u_labels = np.unique(label)
# plotting the results
for i in u_labels:
    plt.scatter(dataVals[label == i, 0] , dataVals[label == i, 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 20, color = 'k')
plt.legend()
plt.show()
# Utilizing a focused subset of the data, frequency of student attendance and highest
# final attempt, the k-means graph shows a strong connection between a student's grade 
# and their class attendance

#using HDBSCAN
import hdbscan
HDB_cluster = hdbscan.HDBSCAN(min_cluster_size=5)
HDB_cluster.fit(dataVals)

plt.scatter(dataVals[:, 0], dataVals[:, 1], c=clusterer.labels_, cmap='rainbow')
plt.title("HDBSCAN Clustering")
plt.show()