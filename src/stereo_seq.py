import cProfile
import pickle
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from umap import UMAP
from matplotlib import pyplot as plt

from src.clustering import cluster_data, cluster_data_custom


def data_analysis(df: pd.DataFrame, eps: float = 70, min_points: float = 10, sim_ind: float = 0.7) -> None:
    processed_data = process_data(df)

    # cluster_data(transformed_data)

    """
    # Optimum finding automation
    
    eps_values = [70, 90, 110, 130, 150]
    min_points_values = [10, 20, 30, 40, 50]
    sim_int_values = [0.6, 0.7, 0.8, 0.9]
    
    for eps_value in eps_values:
        for min_points_value in min_points_values:
            for sim_int_value in sim_int_values:
                print(f'Parameters: {eps_value}, {min_points_value}, {sim_int_value}')
                cluster_data_custom(processed_data, eps_value, min_points_value, sim_int_value)
    """

    cluster_data_custom(processed_data, eps, min_points, sim_ind)


def process_data(df: pd.DataFrame) -> List:
    geneIDs = df['geneID'].drop_duplicates().sort_values(ascending=True).tolist()
    transformed_data = df[['x', 'y', 'cell']].drop_duplicates(subset='cell')

    data = []
    for _, row in transformed_data.iterrows():
        d = [[row['x'], row['y']]]
        gene_dict = dict.fromkeys(geneIDs, 0)
        for _, r in df.loc[df['cell'] == row['cell']].iterrows():
            gene_dict[r['geneID']] = r['MIDCounts']
        d.append(list(gene_dict.values()))
        data.append(d)

    # temp_data = []
    # for t in transformed_data:
    #     temp_data.append(t[0])
    #
    # temp_data = normalize_data(np.array(temp_data))
    # temp_data = temp_data.tolist()
    #
    # for i, t in enumerate(transformed_data):
    #     t[0] = temp_data[i]

    with open('../data/transformed_data', 'wb') as f:
        pickle.dump(data, f)

    # with open('../data/transformed_data', 'rb') as f:
    #     data = pickle.load(f)

    return data


def dimensionality_reduction(df: pd.DataFrame) -> None:
    numeric_data = df[['x', 'y', 'MIDCounts', 'cell']].to_numpy()
    # numeric_data = normalize(numeric_data)
    transformed_data = PCA(n_components=2).fit_transform(numeric_data)
    print(transformed_data.shape)

    # processed_data = process_data(df)
    # processed_data = [data[1] for data in processed_data]
    # transformed_data = PCA(n_components=2).fit_transform(processed_data)

    # plot_2d_plt(transformed_data[:, 0], transformed_data[:, 1])
    plot_2d(transformed_data[:, 0], transformed_data[:, 1])


def plot_2d(component1: List, component2: List) -> None:
    fig = go.Figure(data=go.Scatter(
        x=component1,
        y=component2,
        mode='markers',
        marker=dict(
            color='green',  # set color equal to a variable
            colorscale='Rainbow',  # one of plotly colorscales
            showscale=False,
            line_width=1
        )
    ))
    # fig.update_xaxes(showgrid=False, visible=False)
    # fig.update_yaxes(showgrid=False, visible=False)
    fig.layout.template = 'plotly_dark'

    fig.show()


def plot_2d_plt(component1: List, component2: List) -> None:
    plt.style.use('dark_background')
    plt.axis('off')
    plt.scatter(component1, component2, s=3)
    plt.show()


def normalize_data(data: List) -> List:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    # Code performance check
    # cProfile.run('data_analysis()')

    dataset = pd.read_csv("../data/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin_merge.tsv", sep="\t")
    data_analysis(dataset)
