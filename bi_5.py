# K-Means Clustering on Iris Dataset

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Excel dataset
df = pd.read_csv("Iris.csv")

# Display first 5 rows
print(df.head())

# Remove missing values
df.dropna(inplace=True)

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

# Select features for clusteringpi
X = df[['SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData Standardized Successfully!")
print("Shape:", X_scaled.shape)

# ---------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))

plt.plot(K, inertia, 'bo-')

plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.grid(True)
plt.show()

print("\nOptimal number of clusters is approximately: 3")

# ---------------------------------------------------
# APPLY K-MEANS CLUSTERING
# ---------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Predict clusters
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Display clustered data
print("\nClustered Dataset:")
print(df.head())

# ---------------------------------------------------
# VISUALIZE CLUSTERS
# ---------------------------------------------------

plt.figure(figsize=(8,6))

# Scatter plot
plt.scatter(
    df['SepalLengthCm'],
    df['PetalLengthCm'],
    c=df['Cluster'],
    cmap='viridis',
    s=60
)

# Cluster centers
centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centers[:,0],
    centers[:,2],
    c='red',
    s=200,
    marker='X',
    label='Centroids'
)

# Labels and title
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('K-Means Clustering on Iris Dataset')

plt.legend()
plt.grid(True)

# Show plot
plt.show()