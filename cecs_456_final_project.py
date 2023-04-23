# -*- coding: utf-8 -*-
"""CECS 456 Final Project.ipynb

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('final_grades.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#boxplot of grades
plt.boxplot(y) #shows the score of the students' grades as a boxplot.
plt.show()

#Histogram of the number of grades with a specific score.
plt.hist(y)
plt.show()

# Creating Correlation Matrix
corrMatrix = dataset.corr()

# Plotting Correlation Matrix
plt.matshow(corrMatrix)
plt.show()