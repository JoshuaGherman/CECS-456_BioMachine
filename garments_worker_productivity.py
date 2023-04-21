'''
Tasks: Clustering student behavior based on their test scores 
Instructions/Directions: 
    Perform exploratory data analysis (EDA) using data visualization, for example, histogram of the features, boxplot, and apart from this you are encouraged to explore EDA and plot relevant graphs.   
        1.1 Identify the outliers in the dataset   
        1.2 Plot the correlation matrix for the dataset.   
        1.3 Plot the graphical distribution for the variables   
    Identify the optimal number of clusters in the dataset.
        2.1.  You may want to compare silhouette and elbow method.   
    Use k-means algorithm for creating the clusters.  
        3.1 Interpret each of the clusters in question 3.  
    Use HDBSCAN to perform hierarchical clustering and plot the dendrogram.  
    Compare the results of various clustering algorithms. 
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create a directory to save the figures
if not os.path.exists('figures'):
    os.mkdir('figures')

# Load the dataset
df = pd.read_csv('garments_worker_productivity.csv')  # Replace 'your_dataset.csv' with the actual filename or path of your dataset

# def preprocess_data(df):
# 1.1 Identify the outliers in the dataset
# Plot boxplots for numerical features to identify outliers
numerical_features = df.select_dtypes(include='number').columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=feature, data=df)
    plt.title(f'Boxplot of {feature}')
    plt.savefig(f'figures/boxplot_{feature}.png')
    plt.close()

# 1.2 Plot the correlation matrix for the dataset
# Calculate and plot the correlation matrix
corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('figures/correlation_matrix.png')
plt.close()

# 1.3 Plot the graphical distribution for the variables
# Plot histograms for numerical features
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], bins=20, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(f'figures/histogram_{feature}.png')
    plt.close()

# Plot countplots for categorical features
categorical_features = df.select_dtypes(include='object').columns
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=df)
    plt.title(f'Countplot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.savefig(f'figures/countplot_{feature}.png')
    plt.close()

# preprocess_data(df)