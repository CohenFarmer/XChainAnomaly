import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet("data/datasets/cross_chain_unified.parquet", engine='pyarrow')
shortDf = df.head(500)

pairs = (
    shortDf[["src_from_address", "dst_to_address"]]
    .dropna()
)

G = nx.DiGraph()
for u, v in pairs.itertuples(index=False):
    if G.has_edge(u, v):
        G[u][v]["weight"] = G[u][v].get("weight", 0) + 1
    else:
        G.add_edge(u, v, weight=1, kind="src→dst")

nx.draw(G, with_labels=False, node_color='lightblue', node_size=5)
plt.show()