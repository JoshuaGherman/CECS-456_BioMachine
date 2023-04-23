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

# Bar chart showing frequency with which students attended class
plt.bar([1,2,3,4,5,6], attFreqList)
plt.title('Student Attendance')
plt.xlabel('Number of Sessions Attended')
plt.ylabel('Number of Students')
plt.show()

# Creating Correlation Matrix
corrMatrix = dataset.corr()
#print(corrMatrix)
# Visulization of Correlation Matrix
sn.heatmap(corrMatrix, annot=True)
plt.show()

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
