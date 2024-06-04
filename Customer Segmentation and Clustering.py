#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:/Users/ODUOBUK/Documents/My Data Sources/segmentation Data/Mall_Customers.csv")


# In[3]:


df.head()


# # Univariate Analysis

# In[4]:


df.describe()


# In[5]:


sns.displot(df['Annual Income (k$)'])


# In[6]:


df.columns


# In[7]:


columns = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.displot(df[i])
    


# In[8]:


#sns.kdeplot(df['Annual Income (k$)'],shade=True,hue='Gender')

# Use the correct function call
sns.kdeplot(data=df, x='Annual Income (k$)', shade=True, hue='Gender')


# In[9]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df, x=i, shade=True, hue='Gender')
    plt.title(f'KDE Plot of {i}')
    plt.show()
    


# In[10]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df, x='Gender',y=df[i])
    
    


# In[11]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[12]:


sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[13]:


df=df.drop('CustomerID', axis=1)
sns.pairplot(df, hue='Gender')


# In[14]:


grouped_means = df.groupby('Gender')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(grouped_means)


# In[15]:


numeric_df = df.select_dtypes(include=[float, int])

correlation_matrix = numeric_df.corr()
print(correlation_matrix)


# In[57]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
plt.savefig('Correlation_matrix.png')


# # Clustering - Univariate, Bivariate, Multivariate

# In[17]:


clustering1 = KMeans(n_clusters=6)


# In[18]:


annual_income = df[['Annual Income (k$)']]
# Fit the model
clustering1.fit(annual_income)


# In[19]:


clustering1.labels_


# In[20]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[21]:


df['Income Cluster'].value_counts()


# In[22]:


clustering1.inertia_


# In[23]:


inertia_scores=[]
for i in range(1,11):
    Kmeans=KMeans(n_clusters=i)
    Kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(Kmeans.inertia_)
plt.plot(range(1, 11), inertia_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.show()


# In[24]:


# Group by 'Income Cluster' and calculate mean values
cluster_means = df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

# Display the mean values for each cluster
print(cluster_means)


# In[25]:


#Bivariate Clustering


# In[26]:


clustering2 = KMeans()
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering1.labels_
df.head()


# In[30]:


inertia_scores2 = []

# Loop over a range of cluster numbers to find the optimal number of clusters
for i in range(1, 11):
    # Initialize KMeans with i clusters and a random state for reproducibility
    kmeans2 = KMeans(n_clusters=i, random_state=42)
    
    # Fit KMeans to the data
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    
    # Append the inertia score (sum of squared distances to nearest cluster center)
    inertia_scores2.append(kmeans2.inertia_)

# Plot the inertia scores to visualize the "elbow"
plt.plot(range(1, 11), inertia_scores2, marker='o')
plt.show()


# In[35]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[56]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster', palette='tab10')
plt.savefig('Clustering_bivariate.png')


# In[38]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[39]:


df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[41]:


#Multivariate Clustering
from sklearn.preprocessing import StandardScaler


# In[42]:


scale = StandardScaler()


# In[43]:


df.head()


# In[45]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[46]:


dff.columns


# In[48]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff.head()


# In[50]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[58]:


inertia_scores3 = []

for i in range(1, 11):
    kmeans3 = KMeans(n_clusters=i, random_state=42)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1, 11), inertia_scores3, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.show()


# In[54]:


df.to_csv('Clustering.csv')

