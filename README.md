# Customer-Segmentation-and-Clustering


# Mall Customer Segmentation Analysis

## Introduction
This Python script performs an in-depth analysis of a dataset of mall customers, with the goal of identifying meaningful customer segments that can inform marketing and business strategies. The dataset, "Mall_Customers.csv," contains information about the customers, including their age, annual income, and spending score.

The analysis includes the following steps:

1. Univariate Analysis: Exploring the distribution and characteristics of each individual feature in the dataset, such as age, annual income, and spending score.
2. Bivariate Analysis: Examining the relationships between pairs of features, including visualizations and statistical analysis.
3. Clustering: Applying the K-Means algorithm to identify distinct customer segments based on the annual income and spending score features.

## Libraries Used
The script utilizes the following Python libraries:

1. Pandas: For data manipulation and analysis.
2. Seaborn: For data visualization.
3. Matplotlib: For additional data visualization.
4. Scikit-learn (sklearn): For the K-Means clustering algorithm.

## Data Loading and Preprocessing
The script starts by loading the "Mall_Customers.csv" dataset into a Pandas DataFrame named `df`. No additional preprocessing steps are performed, as the dataset appears to be in a clean and ready-to-use format.

## Univariate Analysis
The univariate analysis section explores the distribution and characteristics of each feature in the dataset:

1. Descriptive Statistics: The script prints the descriptive statistics of the dataset, including the mean, standard deviation, minimum, and maximum values for each feature.
2. Histograms: The script creates histograms for the 'Annual Income (k$)' feature, as well as individual histograms for the 'Age,' 'Annual Income (k$),' and 'Spending Score (1-100)' features.
3. Kernel Density Estimation (KDE) Plots: The script creates KDE plots for the 'Annual Income (k$)' feature, colored by gender, to visualize the distribution of annual income for each gender.

## Bivariate Analysis
The bivariate analysis section explores the relationships between pairs of features in the dataset:

1. Scatter Plot: The script creates a scatter plot of 'Annual Income (k$)' vs 'Spending Score (1-100)' to visualize the relationship between these two features.
2. Pairplot: The script creates a pairplot of the dataset, colored by gender, to explore the relationships between all pairs of features.
3. Mean Values by Gender: The script calculates the mean values for each feature, grouped by gender, and prints the results.
4. Correlation Matrix: The script calculates the correlation matrix for the numeric features and prints the results. It also creates a heatmap of the correlation matrix and saves it as an image file.

## Clustering
The clustering section applies the K-Means algorithm to identify distinct customer segments based on the 'Annual Income (k$)' and 'Spending Score (1-100)' features:

1. K-Means Clustering: The script creates a K-Means clustering model with 6 clusters using the 'Annual Income (k$)' and 'Spending Score (1-100)' features.
2. Cluster Assignment: The script fits the K-Means model to the data and assigns the cluster labels to a new column in the DataFrame.
3. Cluster Sizes: The script prints the number of data points in each cluster.
4. Elbow Curve: The script calculates the inertia (the sum of squared distances of samples to their closest cluster center) for different numbers of clusters and plots the elbow curve to determine the optimal number of clusters.

## Conclusion
This comprehensive analysis of the mall customer dataset provides valuable insights into the characteristics and relationships of the customers, as well as the identification of distinct customer segments using the K-Means clustering algorithm. The results of this analysis can be used to inform marketing strategies, product development, and customer-centric business decisions.
