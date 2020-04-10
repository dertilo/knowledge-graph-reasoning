# TSNE is just for fun! in order to visualize the clusters
import random

import torch
from MulticoreTSNE import MulticoreTSNE as TSNE


from matplotlib import pyplot as plt
from util import data_io


def plot_tsned(X, ent2id: dict):

    norm = plt.Normalize(1, 4)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    sc = ax.scatter(
        X[:, 0],
        X[:, 1],
        s=40,
        cmap=cmap,
        marker="o",
        # norm=norm,
        linewidths=0.0,
    )

    for txt, i in ent2id.items():
        ax.annotate(txt, (X[i][0], X[i][1]))
    plt.savefig("scatterplot.png")
    # plt.show()


if __name__ == "__main__":
    tsne = TSNE(n_components=2, n_jobs=4, n_iter=1000)
    X = torch.load("entity_embeddings.pt")
    ent2id = data_io.read_json("ent2id.json")
    some_entities = {
        k: v
        for k, v in ent2id.items()
        if k
        in [
            "human",
            "animal",
            "organism",
            "vertebrate",
            "bacterium",
            "plant",
            "fungus",
            "virus",
            "mammal",
        ]
    }
    X_embedded = tsne.fit_transform(X)
    idx = list(range(len(ent2id)))
    random.shuffle(idx)
    some_idx = idx[:10]
    some_entities.update({k: v for k, v in ent2id.items() if v in some_idx})
    plot_tsned(X_embedded, some_entities)
