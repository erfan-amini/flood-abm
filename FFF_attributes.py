"""
Flood Adaptation ABM - Agent Attributes Module (v10)

Generates agent attributes and computes Jaccard similarity.
Ref: Jaccard (1912). See User_Manual_v10.docx for details.
"""

import numpy as np


# ============================================================================
# SIMILARITY FUNCTION
# ============================================================================

def jaccard_similarity(attr_a, attr_b):
    """
    Compute Jaccard similarity between two attribute vectors.

    Reference: Jaccard (1912).
    """
    matches = np.sum(np.asarray(attr_a) == np.asarray(attr_b))
    return matches / len(attr_a)


# ============================================================================
# ATTRIBUTE GENERATOR CLASS
# ============================================================================

class AttributeGenerator:
    """Generate categorical attributes for agents."""

    def __init__(self, n_agents, n_attributes, n_classes,
                 enable_heterogeneity, rng=None):
        """
        Parameters
        ----------
        n_agents : int
            Number of agents.
        n_attributes : int
            Attributes per agent.
        n_classes : int
            Possible classes per attribute.
        enable_heterogeneity : bool
            If False, all agents have identical attributes (S=1).
            If True, agents have random attributes.
        rng : np.random.Generator or None
            Random generator for reproducibility.
        """
        self.n_agents = n_agents
        self.n_attributes = n_attributes
        self.n_classes = n_classes
        self.enable_heterogeneity = enable_heterogeneity
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate(self):
        """
        Generate attribute-class assignments.

        Returns
        -------
        attributes : ndarray, shape (n_agents, n_attributes)
        """
        if self.enable_heterogeneity:
            return self.rng.integers(0, self.n_classes,
                                     size=(self.n_agents, self.n_attributes))
        else:
            return np.zeros((self.n_agents, self.n_attributes), dtype=int)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("Attributes Module Test (v10)")
    print("=" * 40)

    rng = np.random.default_rng(42)

    gen_off = AttributeGenerator(n_agents=5, n_attributes=5, n_classes=5,
                                 enable_heterogeneity=False, rng=rng)
    attrs_off = gen_off.generate()
    print(f"\nHeterogeneity OFF:")
    print(f"  Agent 0: {attrs_off[0]}")
    print(f"  Similarity(0,1): {jaccard_similarity(attrs_off[0], attrs_off[1]):.2f}")

    gen_on = AttributeGenerator(n_agents=5, n_attributes=5, n_classes=5,
                                enable_heterogeneity=True, rng=rng)
    attrs_on = gen_on.generate()
    print(f"\nHeterogeneity ON:")
    print(f"  Agent 0: {attrs_on[0]}")
    print(f"  Agent 1: {attrs_on[1]}")
    print(f"  Similarity(0,1): {jaccard_similarity(attrs_on[0], attrs_on[1]):.2f}")
