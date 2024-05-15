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
    percent_changes = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

        if k > 1:
            percent_change = ((inertias[-2] - inertias[-1]) / inertias[-2]) * 100
            percent_changes.append(percent_change)
        else:
            percent_changes.append(0)

    # Generate elbow plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.plot(means, inertias, 'o-', color='tab:blue')
    ax1.tick_params(axis='y')

    # Annotate the percentage changes
    for i in range(1, len(means)):
        ax1.annotate(f'{percent_changes[i]:.1f}%', 
                     xy=(means[i], inertias[i]), 
                     xytext=(0, 10), 
                     textcoords='offset points', 
                     color='red', 
                     fontsize=10)

    plt.grid(True)
    plt.show()

optimise_k_means(data[["feature1", "feature2", "..."]], 10)