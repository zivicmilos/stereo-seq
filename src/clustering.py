import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from random import randint

from src.dbscan import DBScan


def cluster_data(transformed_data):
    db = DBSCAN(eps=0.01, min_samples=100).fit(transformed_data)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = transformed_data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = transformed_data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}, Estimated number of noise points: {n_noise_}")
    plt.show()


def cluster_data_custom(xy, eps, min_points, sim_ind):
    dbscan = DBScan(epsilon=eps, min_points=min_points, similarity_index=sim_ind)
    dbscan.fit(xy)

    print(f'Estimated number of clusters: {len(dbscan.clusters)}')
    print(f'Estimated number of noise points: {len(dbscan.outliers)}')

    colors = []
    n = len(dbscan.clusters)
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    plt.style.use('dark_background')
    for i, cluster in enumerate(dbscan.clusters):
        for d in cluster.data:
            plt.scatter(d[0][0], d[0][1], c=colors[i])

    for outlier in dbscan.outliers:
        plt.scatter(outlier[0][0], outlier[0][1], c='w')

    plt.show()
