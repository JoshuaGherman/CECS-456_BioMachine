import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import hdbscan

# Load the dataset
df = pd.read_csv('garments_worker_productivity.csv')

# Display the first few rows of the dataset
df.head()

# Perform EDA using data visualization

# Histogram of the features
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# Boxplot for the variables
plt.figure(figsize=(10,8))
sns.boxplot(data=df)
plt.title('Boxplot of Variables')
plt.show()

# Correlation matrix
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot of variables
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pairplot of Variables')
plt.show()

