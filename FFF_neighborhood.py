"""
Flood Adaptation ABM - Neighborhood Module (v12)

Identifies spatial neighborhoods from agent positions using DBSCAN
density-based clustering (Ester et al., 1996).

Agents within the same cluster share a neighborhood and are subject
to similarity-based social learning. Agents labeled -1 by DBSCAN
(noise points, typically connectors) belong to no neighborhood.

References:
-----------
Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).
    A density-based algorithm for discovering clusters in large
    spatial databases with noise. KDD-96, 226-231.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def identify_neighborhoods(positions, eps, min_samples=4):
    """
    Assign agents to spatial neighborhoods via DBSCAN (Ester et al., 1996).

    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        Agent (x, y) coordinates.
    eps : float
        Maximum distance between two agents in the same neighborhood.
        Passed directly as DBSCAN eps parameter.
    min_samples : int
        Minimum number of agents (including self) required to form a
        neighborhood core point. Default 4 matches min_cluster_size=3
        neighbors plus the agent itself.

    Returns
    -------
    labels : ndarray, shape (n,)
        Neighborhood label per agent. -1 = no neighborhood (isolated
        or connector agent).
    n_neighborhoods : int
        Number of distinct neighborhoods identified.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(np.asarray(positions))
    n_neighborhoods = len(set(labels) - {-1})
    return labels, n_neighborhoods


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("Neighborhood Module Test (v12)")
    print("=" * 40)

    # Two clusters with a connector agent between them
    positions = np.array([
        [0.1, 0.5], [0.15, 0.5], [0.12, 0.55], [0.08, 0.48],
        [0.25, 0.5],  # connector
        [0.4, 0.5], [0.45, 0.5], [0.42, 0.55], [0.38, 0.48],
    ])
    labels, n = identify_neighborhoods(positions, eps=0.09, min_samples=3)
    print(f"Agents: {len(positions)}")
    print(f"Neighborhoods: {n}")
    print(f"Labels: {labels}")
    print(f"Isolated: {np.sum(labels == -1)}")
