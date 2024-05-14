import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# generating elbow figure to show the most optimum numbers of k
def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    #generate elbow plot
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

optimise_k_means(data[["feature1", "feature2", "..."]], 10)