"""
Flood Adaptation ABM - Network Module (v12)

Builds a social network with binary connections: agents within
DISTANCE_THRESHOLD are connected; agents beyond it are not.
No distance decay is applied to edge weights.

Each edge stores the Jaccard similarity S(i,j) between the two
agents' attribute vectors (Jaccard, 1912), used by the similarity-
based learning channel in the main model.

References:
-----------
Jaccard (1912), The distribution of the flora in the alpine zone.
McPherson et al. (2001), Birds of a feather: Homophily in social networks.
"""

import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial.distance import cdist
from FFF_attributes import jaccard_similarity


class NetworkBuilder:
    """
    Build a binary social network from agent positions.

    Nodes = agents (integer IDs 0..n-1).
    An edge exists between i and j if d(i,j) <= distance_threshold.
    Each edge stores:
      - distance: Euclidean distance between agents
      - similarity: Jaccard similarity S(i,j) (Jaccard, 1912)

    Uses scipy.spatial.distance.cdist for vectorized distance
    computation and numpy broadcasting for vectorized Jaccard
    similarity.
    """

    def __init__(self, positions, attributes,
                 distance_threshold, user_edges_csv=None):
        self.positions = np.asarray(positions)
        self.attributes = np.asarray(attributes)
        self.n_agents = len(positions)
        self.distance_threshold = distance_threshold
        self.user_edges_csv = user_edges_csv

    def build(self):
        """Construct the network with binary connections."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_agents))

        # Vectorized pairwise distances (scipy cdist)
        dist_matrix = cdist(self.positions, self.positions)

        # Upper-triangle pairs within threshold
        rows, cols = np.where(
            (dist_matrix <= self.distance_threshold) & (dist_matrix > 0))
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]

        # Distances and Jaccard similarity for all candidate edges
        d_all = dist_matrix[rows, cols]
        attr_i = self.attributes[rows]
        attr_j = self.attributes[cols]
        S_all = np.mean(attr_i == attr_j, axis=1)

        # Add edges (binary connection, no distance decay)
        for idx in range(len(rows)):
            G.add_edge(int(rows[idx]), int(cols[idx]),
                       distance=float(d_all[idx]),
                       similarity=float(S_all[idx]),
                       edge_type="spatial")

        # User-defined edges
        if self.user_edges_csv is not None:
            df = pd.read_csv(self.user_edges_csv)
            for _, row in df.iterrows():
                i, j = int(row["source"]), int(row["target"])
                if i == j or G.has_edge(i, j):
                    continue
                d = dist_matrix[i, j]
                S = jaccard_similarity(self.attributes[i], self.attributes[j])
                G.add_edge(i, j, distance=d,
                           similarity=S, edge_type="user_defined")

        return G


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("Network Module Test (v12 - binary connections)")
    print("=" * 50)

    positions = np.array([[0.1, 0.5], [0.15, 0.5], [0.2, 0.5],
                          [0.5, 0.5], [0.55, 0.5], [0.6, 0.5]])
    attributes = np.array([[0, 1], [0, 1], [1, 0],
                           [0, 1], [0, 1], [1, 1]])

    builder = NetworkBuilder(positions, attributes, distance_threshold=0.09)
    G = builder.build()

    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    for u, v, data in G.edges(data=True):
        print(f"  Edge ({u},{v}): dist={data['distance']:.3f}, "
              f"sim={data['similarity']:.1f}")
