import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime


# setting up the data, change to your respective file path
users = pd.read_excel("filepath", sheet_name="sheetname")


# selecting features that matters, change to your preferred features
features = ["feature1", "feature2", "..."]


# ensure there is no na
users = users.dropna(subset=features)


# make a copy of the underlying data
data = users[features].copy()


# scale the data hence no column dominates other
data = (data - data.min()) / (data.max() - data.min()) * 9 + 1


# setting up the iterations & clusters, change to suit your preferred analysis
max_iterations = 100
k = 4
iteration = 1

# initialize random centroids
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis = 1)

centroids = random_centroids(data, k)


# label each data point
def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


labels = get_labels(data, centroids)


# update centroids based on who is in the clusters by using geometric means
def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


# plotting data to the centroids
def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration of {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1], marker='x', c='red')
    plt.show()


centroids = random_centroids(data, k)
old_centroids = pd.DataFrame()

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids

    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, k)
    # activate to output the iteration plot
    # plot_clusters(data, labels, centroids, iteration)
    iteration += 1


# checking the centroids
centroids


# extracting customers to labels
now = datetime.now().isoformat(timespec='seconds')

output = pd.concat([users, pd.DataFrame(labels, columns=['Cluster'])], axis=1).to_csv(f"kmeans_output-{now}")

output