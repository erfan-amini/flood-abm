"""
Flood Adaptation ABM v12 - Bayesian Belief Updating (Three Channels)

Binary hypothesis model:
  H1 = "my situation warrants retrofitting"
  H0 = "my situation does not warrant retrofitting"

Agents update P(H1) via Bayes' theorem in odds form
(Jaynes, 2003, Ch. 4; Kass & Raftery, 1995):
  posterior_odds = prior_odds x Bayes_factor

Three evidence channels:
  1. Personal flood experience (LAMBDA_FLOOD):
     Flood events multiply odds; safe years produce no update
     (availability heuristic, Tversky & Kahneman, 1974).
  2. Proximity-based social learning (LAMBDA_SOCIAL):
     Binary connection: connected neighbors who retrofit deliver
     the full Bayes factor (McPherson et al., 2001).
  3. Similarity-based social learning (LAMBDA_SIMILARITY):
     Within DBSCAN-identified neighborhoods (Ester et al., 1996),
     Jaccard attribute similarity (Jaccard, 1912) scales the factor:
     effective = LAMBDA_SIMILARITY ^ S(i,j).

References:
-----------
Jaynes (2003), Probability Theory: The Logic of Science, Ch. 4.
Kass & Raftery (1995), Bayes Factors, JASA, 90(430), 773-795.
Tversky & Kahneman (1974), Judgment under Uncertainty.
Rogers (1975), A protection motivation theory of fear appeals.
McPherson et al. (2001), Birds of a feather: Homophily.
Jaccard (1912), Distribution of the flora in the alpine zone.
Ester et al. (1996), A density-based algorithm (DBSCAN), KDD-96.

See User_Manual_v12.docx for full documentation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime

import mesa
from mesa.space import NetworkGrid

from FFF_flood import FloodGenerator
from FFF_spatial import SpatialGenerator
from FFF_attributes import AttributeGenerator
from FFF_network import NetworkBuilder
from FFF_neighborhood import identify_neighborhoods


# ============================================================================
# USER PARAMETERS
# ============================================================================

# --- Simulation ---
TIME_STEPS = 100
RANDOM_SEED = 42

# --- Spatial Layout & Population ---
SPATIAL_MODE = 2               # 0=CSV, 1=random, 2=grid/connected
N_AGENTS = 200                 # Total agents (auto-adjusted in mode 2)
GRID_ROWS = 3                  # Neighborhood grid rows (mode 2)
GRID_COLS = 4                  # Neighborhood grid columns (mode 2)
N_CONNECTORS = 2               # Bridge agents between neighborhoods (0=plain grid)
SLOPE = 0.2                    # Land elevation gradient (elevation = slope * x)
NOISE_FACTOR = 0.05            # Elevation noise as fraction of slope

# --- Agent Attributes (Jaccard, 1912; McPherson et al., 2001) ---
ENABLE_HETEROGENEITY = True    # False = all identical (S=1 for all pairs)
N_ATTRIBUTES = 2               # Attributes per agent
N_CLASSES = 3                  # Categories per attribute

# --- Network (binary connections within distance threshold) ---
DISTANCE_THRESHOLD = 0.09      # Max distance for edges (binary: connected or not)
USER_EDGES_CSV = None          # Optional CSV with (source, target) columns
CSV_PATH = None                # Optional CSV with (x, y, z) for SPATIAL_MODE=0

# --- Neighborhood Identification (Ester et al., 1996) ---
DBSCAN_MIN_SAMPLES = 4         # Min agents (incl. self) to form a neighborhood core

# --- Bayesian Belief Updating — Odds Form (Jaynes, 2003, Ch. 4) ---
# Agent belief = P(H1) where H1 = "my situation warrants retrofitting"
# Updated via: posterior_odds = prior_odds x Bayes_factor
# (Kass & Raftery, 1995)
INITIAL_BELIEF = 0.05          # Prior P(H1): mild awareness in flood zone

# --- Channel 1: Personal Flood Experience (Jaynes, 2003, Ch. 4) ---
# Asymmetric updating (Tversky & Kahneman, 1974): only flood events
# shift belief; safe years leave belief unchanged.
LAMBDA_FLOOD = 1.20            # Bayes factor per flood event

# --- Channel 2: Proximity-Based Social Learning ---
# Binary: connected neighbors (within DISTANCE_THRESHOLD) who retrofit
# deliver the full Bayes factor. No distance weighting.
LAMBDA_SOCIAL = 1.50           # Bayes factor per connected neighbor retrofit

# --- Channel 3: Similarity-Based Social Learning (Jaccard, 1912) ---
# Within DBSCAN neighborhoods (Ester et al., 1996), attribute similarity
# scales the Bayes factor: effective = LAMBDA_SIMILARITY ^ S(i,j).
# S=1 (identical attributes) gives full factor; S=0 gives no update.
LAMBDA_SIMILARITY = 3.00       # Bayes factor at full similarity (S=1)

# --- Protection Motivation Theory (Rogers, 1975) ---
# Heterogeneous thresholds: truncated Normal distribution
PMT_THRESHOLD_MEAN = 0.50      # Population mean threshold
PMT_THRESHOLD_STD = 0.00       # Standard deviation (0 = homogeneous)
PMT_THRESHOLD_LOW = 0.50       # Hard lower bound (truncation)
PMT_THRESHOLD_HIGH = 0.50      # Hard upper bound (truncation)

# --- Flood (GEV - Coles, 2001) ---
RETURN_PERIODS = [10, 20, 50, 100]
FLOOD_LEVELS = [0.05, 0.10, 0.15, 0.30]

# --- Observed Data for Comparison ---
OBSERVED_BINS = ["0", "1", "2-3", "4+"]
OBSERVED_RATES = [13, 18, 27, 57]

# --- Visualization ---
FIG_DPI = 300
FONT_FAMILY = "Palatino Linotype"
FONT_SIZE = 16


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_output_dir():
    """Create timestamped output directory next to this script."""
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = script_dir / "output" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_parameters(output_dir):
    """Save all model parameters to text file."""
    path = os.path.join(output_dir, "parameters.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("FLOOD ADAPTATION ABM v12 - PARAMETER VALUES\n")
        f.write("=" * 60 + "\n\n")

        f.write("SIMULATION\n" + "-" * 40 + "\n")
        f.write(f"TIME_STEPS             = {TIME_STEPS}\n")
        f.write(f"RANDOM_SEED            = {RANDOM_SEED}\n\n")

        f.write("SPATIAL LAYOUT & POPULATION\n" + "-" * 40 + "\n")
        f.write(f"SPATIAL_MODE           = {SPATIAL_MODE}\n")
        f.write(f"N_AGENTS               = {N_AGENTS}\n")
        f.write(f"GRID_ROWS              = {GRID_ROWS}\n")
        f.write(f"GRID_COLS              = {GRID_COLS}\n")
        f.write(f"N_CONNECTORS           = {N_CONNECTORS}\n")
        f.write(f"SLOPE                  = {SLOPE}\n")
        f.write(f"NOISE_FACTOR           = {NOISE_FACTOR}\n\n")

        f.write("AGENT ATTRIBUTES (Jaccard, 1912)\n" + "-" * 40 + "\n")
        f.write(f"ENABLE_HETEROGENEITY   = {ENABLE_HETEROGENEITY}\n")
        f.write(f"N_ATTRIBUTES           = {N_ATTRIBUTES}\n")
        f.write(f"N_CLASSES              = {N_CLASSES}\n\n")

        f.write("NETWORK (binary connections)\n" + "-" * 40 + "\n")
        f.write(f"DISTANCE_THRESHOLD     = {DISTANCE_THRESHOLD}\n")
        f.write(f"USER_EDGES_CSV         = {USER_EDGES_CSV}\n\n")

        f.write("NEIGHBORHOODS (Ester et al., 1996)\n" + "-" * 40 + "\n")
        f.write(f"DBSCAN_MIN_SAMPLES     = {DBSCAN_MIN_SAMPLES}\n\n")

        f.write("BAYESIAN BELIEF - THREE CHANNELS (Jaynes, 2003)\n" + "-" * 40 + "\n")
        f.write(f"INITIAL_BELIEF         = {INITIAL_BELIEF}\n")
        f.write(f"LAMBDA_FLOOD           = {LAMBDA_FLOOD}  (Ch.1: personal)\n")
        f.write(f"LAMBDA_SOCIAL          = {LAMBDA_SOCIAL}  (Ch.2: proximity)\n")
        f.write(f"LAMBDA_SIMILARITY      = {LAMBDA_SIMILARITY}  (Ch.3: similarity)\n\n")

        f.write("PMT THRESHOLD (Rogers, 1975)\n" + "-" * 40 + "\n")
        f.write(f"PMT_THRESHOLD_MEAN     = {PMT_THRESHOLD_MEAN}\n")
        f.write(f"PMT_THRESHOLD_STD      = {PMT_THRESHOLD_STD}\n")
        f.write(f"PMT_THRESHOLD_LOW      = {PMT_THRESHOLD_LOW}\n")
        f.write(f"PMT_THRESHOLD_HIGH     = {PMT_THRESHOLD_HIGH}\n\n")

        f.write("FLOOD (GEV - Coles, 2001)\n" + "-" * 40 + "\n")
        f.write(f"RETURN_PERIODS         = {RETURN_PERIODS}\n")
        f.write(f"FLOOD_LEVELS           = {FLOOD_LEVELS}\n")
    print(f"Saved: {path}")


def parse_flood_bins(bin_labels):
    """
    Parse bin label strings into (min, max) tuples.
    Supports: "0", "1", "2-3", "4+".
    """
    ranges = []
    for label in bin_labels:
        if "+" in label:
            ranges.append((int(label.replace("+", "")), float("inf")))
        elif "-" in label:
            parts = label.split("-")
            ranges.append((int(parts[0]), int(parts[1])))
        else:
            n = int(label)
            ranges.append((n, n))
    return ranges


# Pre-parse observed bins
FLOOD_BIN_RANGES = parse_flood_bins(OBSERVED_BINS)


def categorize_flood_count(count):
    """Categorize flood count into a bin label from OBSERVED_BINS."""
    for (lo, hi), label in zip(FLOOD_BIN_RANGES, OBSERVED_BINS):
        if lo <= count <= hi:
            return label
    return OBSERVED_BINS[-1]


# ============================================================================
# BAYESIAN UPDATE — ODDS FORM (Jaynes, 2003, Ch. 4; Kass & Raftery, 1995)
# ============================================================================

def bayesian_update(belief, bayes_factor):
    """
    Bayesian belief update in odds form (Jaynes, 2003, Ch. 4).

    posterior_odds = prior_odds × bayes_factor
    Then convert back to probability.

    This is algebraically equivalent to the standard form of Bayes'
    theorem. The Bayes factor is the ratio P(E|H1)/P(E|H0), which
    measures how much the evidence favors H1 over H0
    (Kass & Raftery, 1995).

    Parameters
    ----------
    belief : float
        Current P(H1) before observing evidence.
    bayes_factor : float
        P(evidence|H1) / P(evidence|H0). The odds multiplier.

    Returns
    -------
    float
        Updated P(H1) after observing evidence.
    """
    odds = belief / (1.0 - belief)
    odds *= bayes_factor
    return odds / (1.0 + odds)


# ============================================================================
# HOUSEHOLD AGENT
# ============================================================================

class HouseholdAgent(mesa.Agent):
    """
    Bayesian household agent (Jaynes, 2003).

    Maintains P(H1) where H1 = "my situation warrants retrofitting."
    Three evidence channels update belief via odds-form Bayes factors:
      1. Flood: odds *= LAMBDA_FLOOD (Tversky & Kahneman, 1974)
      2. Proximity: odds *= LAMBDA_SOCIAL per connected neighbor
      3. Similarity: odds *= LAMBDA_SIMILARITY^S(i,j) within neighborhood
    Retrofits when P(H1) >= individual pmt_threshold (Rogers, 1975).
    """

    def __init__(self, model, x, y, z, attributes,
                 initial_belief, pmt_threshold, neighborhood_id):
        super().__init__(model)
        self.x = x
        self.y = y
        self.z = z
        self.belief = initial_belief       # P(H1): probability situation warrants retrofit
        self.pmt_threshold = pmt_threshold # Individual decision threshold
        self.neighborhood_id = neighborhood_id  # DBSCAN cluster label (-1 = none)
        self.is_retrofitted = False
        self.retrofit_step = None
        self.flood_count = 0
        self.attributes = attributes
        # Track which neighbors have already been observed as retrofitted
        self.observed_retrofitted = set()

    def experience_flood(self, flood_level):
        """
        Asymmetric Bayesian update from flood experience.

        Motivated by the availability heuristic (Tversky & Kahneman, 1974):
        vivid flood events multiply odds by LAMBDA_FLOOD; safe years
        produce no update (non-events are not psychologically salient).

        If flooded: odds *= LAMBDA_FLOOD
        If not flooded: no update.
        """
        if self.is_retrofitted:
            return False
        if flood_level > self.z:
            self.flood_count += 1
            self.belief = bayesian_update(self.belief, LAMBDA_FLOOD)
            return True
        return False

    def social_learning(self):
        """
        Two-channel social Bayesian update (one-shot per neighbor).

        For each connected neighbor newly observed as retrofitted:
          Channel 2 — Proximity (McPherson et al., 2001):
            odds *= LAMBDA_SOCIAL (full factor, binary connection)
          Channel 3 — Similarity (Jaccard, 1912):
            if both agents share the same DBSCAN neighborhood:
              S = Jaccard similarity between attribute vectors
              odds *= LAMBDA_SIMILARITY ^ S
            if in different neighborhoods or either is isolated (-1):
              no similarity update
        """
        if self.is_retrofitted:
            return
        for neighbor in self.model.grid.get_neighbors(self.pos, include_center=False):
            if neighbor.is_retrofitted and neighbor.unique_id not in self.observed_retrofitted:
                self.observed_retrofitted.add(neighbor.unique_id)

                # Channel 2: proximity — full Bayes factor for being connected
                self.belief = bayesian_update(self.belief, LAMBDA_SOCIAL)

                # Channel 3: similarity — only within same neighborhood
                same_neighborhood = (
                    self.neighborhood_id >= 0
                    and self.neighborhood_id == neighbor.neighborhood_id)
                if same_neighborhood:
                    S = self.model.G.edges[self.pos, neighbor.pos]["similarity"]
                    if S > 0:
                        effective_factor = LAMBDA_SIMILARITY ** S
                        self.belief = bayesian_update(self.belief, effective_factor)

    def make_decision(self):
        """Retrofit when belief >= individual threshold (Rogers, 1975)."""
        if not self.is_retrofitted and self.belief >= self.pmt_threshold:
            self.is_retrofitted = True
            self.retrofit_step = self.model.current_step

    def step(self):
        self.social_learning()
        self.make_decision()


# ============================================================================
# MODEL CLASS
# ============================================================================

class FloodAdaptationModel(mesa.Model):
    """Flood adaptation with three-channel Bayesian updating (Kazil et al., 2020)."""

    def __init__(self, n_agents=N_AGENTS, time_steps=TIME_STEPS, seed=RANDOM_SEED):
        super().__init__(seed=seed)
        self.n_agents = n_agents
        self.time_steps = time_steps
        self.current_step = 0
        self.current_flood_level = 0.0
        self.flood_history = []

        self._init_components()
        self.agent_data = []
        self.model_data = []

    def _init_components(self):
        """Initialize flood, spatial, attributes, neighborhoods, network, agents."""
        self.flood_generator = FloodGenerator(
            return_periods=RETURN_PERIODS, flood_levels=FLOOD_LEVELS, rng=self.rng)

        spatial_gen = SpatialGenerator(
            n_agents=self.n_agents, mode=SPATIAL_MODE,
            distance_threshold=DISTANCE_THRESHOLD,
            grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
            n_connectors=N_CONNECTORS,
            slope=SLOPE, noise_factor=NOISE_FACTOR,
            csv_path=CSV_PATH, rng=self.rng)
        self.positions, self.elevations = spatial_gen.generate()

        attr_gen = AttributeGenerator(
            n_agents=self.n_agents, n_attributes=N_ATTRIBUTES,
            n_classes=N_CLASSES, enable_heterogeneity=ENABLE_HETEROGENEITY,
            rng=self.rng)
        self.attributes = attr_gen.generate()

        # Identify neighborhoods via DBSCAN (Ester et al., 1996)
        self.neighborhood_labels, self.n_neighborhoods = identify_neighborhoods(
            self.positions, eps=DISTANCE_THRESHOLD, min_samples=DBSCAN_MIN_SAMPLES)

        net_builder = NetworkBuilder(
            positions=self.positions, attributes=self.attributes,
            distance_threshold=DISTANCE_THRESHOLD, user_edges_csv=USER_EDGES_CSV)
        self.G = net_builder.build()
        self.grid = NetworkGrid(self.G)

        # Create agents with initial belief, thresholds, and neighborhood IDs
        self.agents_by_node = {}
        for i in range(self.n_agents):
            thr = np.clip(
                self.rng.normal(PMT_THRESHOLD_MEAN, PMT_THRESHOLD_STD),
                PMT_THRESHOLD_LOW, PMT_THRESHOLD_HIGH)
            agent = HouseholdAgent(
                model=self, x=self.positions[i, 0], y=self.positions[i, 1],
                z=self.elevations[i], attributes=self.attributes[i],
                initial_belief=INITIAL_BELIEF, pmt_threshold=thr,
                neighborhood_id=int(self.neighborhood_labels[i]))
            self.grid.place_agent(agent, i)
            self.agents_by_node[i] = agent

    def step(self):
        self.current_step += 1
        self.current_flood_level = self.flood_generator.sample()
        self.flood_history.append(self.current_flood_level)
        for agent in self.agents:
            agent.experience_flood(self.current_flood_level)
        self.agents.shuffle_do("step")
        self._collect_data()

    def _collect_data(self):
        for agent in self.agents:
            self.agent_data.append({
                "Step": self.current_step, "AgentID": agent.unique_id,
                "x": agent.x, "y": agent.y, "z": agent.z,
                "belief": agent.belief,
                "pmt_threshold": agent.pmt_threshold,
                "neighborhood_id": agent.neighborhood_id,
                "is_retrofitted": agent.is_retrofitted,
                "flood_count": agent.flood_count,
                "retrofit_step": agent.retrofit_step})
        n_ret = sum(1 for a in self.agents if a.is_retrofitted)
        self.model_data.append({
            "Step": self.current_step,
            "flood_level": self.current_flood_level,
            "n_retrofitted": n_ret,
            "pct_retrofitted": 100 * n_ret / self.n_agents,
            "mean_belief": np.mean([a.belief for a in self.agents])})

    def run(self):
        for i in range(self.time_steps):
            self.step()
            if (i + 1) % 20 == 0:
                n_ret = sum(1 for a in self.agents if a.is_retrofitted)
                print(f"  Step {i+1}: Retrofitted = {n_ret}/{self.n_agents}")

    def get_agent_dataframe(self):
        return pd.DataFrame(self.agent_data).set_index(["Step", "AgentID"])

    def get_model_dataframe(self):
        return pd.DataFrame(self.model_data).set_index("Step")


# ============================================================================
# VISUALIZATION
# ============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FONT_FAMILY, "STIXGeneral", "Palatino", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2, "legend.fontsize": FONT_SIZE - 2})


def draw_edges(ax, model, alpha=0.25):
    """Draw network edges (binary connections)."""
    segs = []
    for u, v in model.G.edges():
        a_u, a_v = model.agents_by_node[u], model.agents_by_node[v]
        segs.append([(a_u.x, a_u.y), (a_v.x, a_v.y)])
    if segs:
        ax.add_collection(LineCollection(
            segs, linewidths=0.5, colors="gray", alpha=alpha, zorder=1))


def plot_adoption_curve(model, output_dir):
    df = model.get_model_dataframe()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["pct_retrofitted"], "b-", linewidth=2)
    ax.fill_between(df.index, df["pct_retrofitted"], alpha=0.15)
    ax.set(xlabel="Time Step", ylabel="Retrofitted (%)",
           title="Retrofit Adoption Over Time", xlim=(1, model.time_steps), ylim=(0, 100))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "adoption_curve.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_adoption_by_elevation(model, output_dir):
    agents = list(model.agents)
    terciles = np.percentile([a.z for a in agents], [33.33, 66.67])
    groups = {"Low": [], "Medium": [], "High": []}
    for agent in agents:
        if agent.z <= terciles[0]: groups["Low"].append(agent)
        elif agent.z <= terciles[1]: groups["Medium"].append(agent)
        else: groups["High"].append(agent)
    rates, labels = [], []
    for label, group in groups.items():
        if group:
            rates.append(100 * sum(1 for a in group if a.is_retrofitted) / len(group))
            labels.append(f"{label}\n(n={len(group)})")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, rates, color=["#d62728","#ff7f0e","#2ca02c"], edgecolor="black")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{rate:.1f}%", ha="center", va="bottom")
    ax.set(xlabel="Elevation Tercile", ylabel="Retrofit Rate (%)",
           title="Retrofit Adoption by Elevation",
           ylim=(0, max(rates)*1.2 if rates else 100))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "adoption_by_elevation.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_flood_history(model, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = range(1, len(model.flood_history) + 1)
    ax.bar(steps, model.flood_history, color="steelblue", alpha=0.7, edgecolor="black")
    m = np.mean(model.flood_history)
    ax.axhline(y=m, color="red", linestyle="--", linewidth=2, label=f"Mean = {m:.3f}")
    ax.set(xlabel="Time Step", ylabel="Flood Level",
           title="Annual Flood Levels (GEV Distribution, Coles 2001)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "flood_history.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_network(model, output_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    agents = list(model.agents)
    draw_edges(ax, model, alpha=0.3)
    adopted = [a for a in agents if a.is_retrofitted]
    not_adopted = [a for a in agents if not a.is_retrofitted]
    if not_adopted:
        ax.scatter([a.x for a in not_adopted], [a.y for a in not_adopted],
                   c="lightgray", s=200, edgecolor="black", linewidth=1, zorder=2)
        for a in not_adopted:
            ax.text(a.x, a.y, str(a.flood_count), ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=3)
    if adopted:
        sc = ax.scatter([a.x for a in adopted], [a.y for a in adopted],
            c=[a.retrofit_step for a in adopted], cmap="YlGn", s=200,
            edgecolor="black", linewidth=1, zorder=2, vmin=1, vmax=model.current_step)
        plt.colorbar(sc, ax=ax, shrink=0.6).set_label("Retrofit Step")
        for a in adopted:
            ax.text(a.x, a.y, str(a.flood_count), ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=3)
    ax.set(xlabel="x", ylabel="y",
           title="Social Network (binary connections within threshold)",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Patch(facecolor="lightgray", edgecolor="black", label="Not Retrofitted"),
        Patch(facecolor="#31a354", edgecolor="black", label="Retrofitted")],
        loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "network.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_comparison_observed(model, output_dir):
    agents = list(model.agents)
    bins = {b: {"total": 0, "retrofitted": 0} for b in OBSERVED_BINS}
    for agent in agents:
        cat = categorize_flood_count(agent.flood_count)
        bins[cat]["total"] += 1
        if agent.is_retrofitted: bins[cat]["retrofitted"] += 1
    model_rates = [100*bins[b]["retrofitted"]/bins[b]["total"]
                   if bins[b]["total"]>0 else 0 for b in OBSERVED_BINS]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(OBSERVED_BINS)); width = 0.35
    bars_m = ax.bar(x-width/2, model_rates, width, label="Model",
                    color="steelblue", edgecolor="black")
    bars_o = ax.bar(x+width/2, OBSERVED_RATES, width, label="Observed",
                    color="coral", edgecolor="black")
    for bar, rate in zip(bars_m, model_rates):
        if rate > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"{rate:.0f}%", ha="center", va="center",
                    fontweight="bold", color="white", fontsize=FONT_SIZE-2)
    for bar, rate in zip(bars_o, OBSERVED_RATES):
        if rate > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"{rate}%", ha="center", va="center",
                    fontweight="bold", color="white", fontsize=FONT_SIZE-2)
    ax.set(xlabel="Flood Count Category", ylabel="Retrofit Rate (%)",
           title="Model vs Observed Retrofit Rates")
    ax.set_xticks(x); ax.set_xticklabels(OBSERVED_BINS)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_observed.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_spatial_retrofit_map(model, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    agents = list(model.agents)
    draw_edges(ax, model, alpha=0.2)
    sc = ax.scatter([a.x for a in agents], [a.y for a in agents],
                    c=[a.z for a in agents], cmap="terrain", s=150, alpha=0.7,
                    edgecolor="black", linewidth=0.5, zorder=2)
    for a in agents:
        if a.is_retrofitted:
            ax.scatter(a.x, a.y, c="green", s=150, edgecolor="black",
                       linewidth=1.5, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8).set_label("Elevation")
    ax.set(xlabel="x", ylabel="y",
           title="Spatial Distribution: Elevation and Retrofit Status",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax.legend(handles=[
        Patch(facecolor="green", edgecolor="black", label="Retrofitted"),
        Patch(facecolor="lightgray", edgecolor="black", label="Not Retrofitted")],
        loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "spatial_retrofit_map.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_similarity_distribution(model, output_dir):
    """Distribution of Jaccard similarities on network edges (Jaccard, 1912)."""
    sims = [d["similarity"] for _, _, d in model.G.edges(data=True)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(sims), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(sims):.3f}")
    ax.set(xlabel="Jaccard Similarity S(i,j)", ylabel="Frequency",
           title="Distribution of Attribute Similarity (Jaccard, 1912)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_distribution.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_belief_evolution(model, output_dir):
    """
    Plot Bayesian belief P(H1) evolution (Jaynes, 2003).

    Left panel: mean belief with 10th-90th percentile band.
    Right panel: belief distribution across agents at final step.
    """
    df_model = model.get_model_dataframe()
    df_agent = model.get_agent_dataframe()
    steps = df_model.index.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean belief over time with percentile band
    p10, p90 = [], []
    for step in steps:
        step_beliefs = df_agent.xs(step, level="Step")["belief"].values
        p10.append(np.percentile(step_beliefs, 10))
        p90.append(np.percentile(step_beliefs, 90))

    ax1.fill_between(steps, p10, p90, alpha=0.2, color="blue",
                     label="10th\u201390th percentile")
    ax1.plot(steps, df_model["mean_belief"], "b-", linewidth=2, label="Mean $P(H_1)$")

    # Show threshold range (10th-90th percentile of agent thresholds)
    final_step = steps[-1]
    final_data = df_agent.xs(final_step, level="Step")
    thr_vals = final_data["pmt_threshold"].values
    thr_p10, thr_p90 = np.percentile(thr_vals, [10, 90])
    thr_mean = thr_vals.mean()
    ax1.axhspan(thr_p10, thr_p90, color="red", alpha=0.08,
                label=f"Threshold 10th\u201390th: [{thr_p10:.2f}, {thr_p90:.2f}]")
    ax1.axhline(y=thr_mean, color="red", linestyle="--",
                label=f"Mean threshold = {thr_mean:.2f}")
    ax1.axhline(y=INITIAL_BELIEF, color="gray", linestyle=":",
                label=f"Prior = {INITIAL_BELIEF}")
    ax1.set(xlabel="Time Step", ylabel="$P(H_1)$",
            title="Bayesian Belief Evolution")
    ax1.legend(fontsize=FONT_SIZE-4); ax1.grid(True, alpha=0.3)

    # Right: histogram of final beliefs with threshold range
    final_beliefs = df_agent.xs(steps[-1], level="Step")["belief"].values
    ax2.hist(final_beliefs, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax2.axvspan(thr_p10, thr_p90, color="red", alpha=0.08,
                label=f"Threshold 10th\u201390th: [{thr_p10:.2f}, {thr_p90:.2f}]")
    ax2.axvline(x=thr_mean, color="red", linestyle="--", linewidth=2,
                label=f"Mean threshold = {thr_mean:.2f}")
    ax2.set(xlabel="$P(H_1)$", ylabel="Number of Agents",
            title=f"Final Belief Distribution (Step {steps[-1]})")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "belief_evolution.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def create_animation(model, output_dir, frame_interval=1):
    agent_data = model.get_agent_dataframe()
    all_steps = sorted(agent_data.index.get_level_values("Step").unique())
    sampled_steps = all_steps[::frame_interval]
    if all_steps[-1] not in sampled_steps:
        sampled_steps = list(sampled_steps) + [all_steps[-1]]
    nodes = sorted(model.agents_by_node.keys())
    xs = np.array([model.agents_by_node[n].x for n in nodes])
    ys = np.array([model.agents_by_node[n].y for n in nodes])
    uids = [model.agents_by_node[n].unique_id for n in nodes]
    fig, ax = plt.subplots(figsize=(10, 8))
    def update(frame_idx):
        ax.clear()
        step = sampled_steps[frame_idx]
        step_data = agent_data.xs(step, level="Step")
        draw_edges(ax, model, alpha=0.2)
        is_ret = np.array([step_data.loc[uid, "is_retrofitted"] for uid in uids])
        floods = np.array([int(step_data.loc[uid, "flood_count"]) for uid in uids])
        ax.scatter(xs, ys, c=np.where(is_ret, "green", "lightgray"),
                   s=300, edgecolors="black", linewidth=1, zorder=2)
        for i, (x, y, fc) in enumerate(zip(xs, ys, floods)):
            ax.text(x, y, str(fc), ha="center", va="center",
                    fontsize=6, fontweight="bold", zorder=3)
        ax.set_title(f"Step {step}: Retrofitted = {int(is_ret.sum())}/{model.n_agents}")
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[
            Patch(facecolor="green", edgecolor="black", label="Retrofitted"),
            Patch(facecolor="lightgray", edgecolor="black", label="Not Retrofitted")],
            loc="upper right")
    anim = FuncAnimation(fig, update, frames=len(sampled_steps), interval=300)
    path = os.path.join(output_dir, "spatial_animation.gif")
    anim.save(path, writer=PillowWriter(fps=3)); plt.close()
    print(f"Saved: {path} ({len(sampled_steps)} frames)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FLOOD ADAPTATION ABM v12 - THREE-CHANNEL BAYESIAN UPDATING")
    print("=" * 70)
    print(f"\nSpatial: mode={SPATIAL_MODE}, N={N_AGENTS}, "
          f"grid={GRID_ROWS}x{GRID_COLS}, connectors={N_CONNECTORS}")
    print(f"Network: threshold={DISTANCE_THRESHOLD}, connections=binary")
    print(f"Neighborhoods: DBSCAN eps={DISTANCE_THRESHOLD}, "
          f"min_samples={DBSCAN_MIN_SAMPLES}")
    print(f"Bayesian: P(H1)_0={INITIAL_BELIEF}")
    print(f"  Ch.1 flood:      lambda={LAMBDA_FLOOD}")
    print(f"  Ch.2 proximity:  lambda={LAMBDA_SOCIAL}")
    print(f"  Ch.3 similarity: lambda={LAMBDA_SIMILARITY}")
    print(f"PMT threshold: N({PMT_THRESHOLD_MEAN},{PMT_THRESHOLD_STD}) "
          f"clipped to [{PMT_THRESHOLD_LOW}, {PMT_THRESHOLD_HIGH}]")
    print("-" * 70)

    output_dir = get_output_dir()
    print(f"\nOutput: {output_dir}")
    save_parameters(output_dir)

    print("\nRunning simulation...")
    model = FloodAdaptationModel()
    model.run()

    agents = list(model.agents)
    n_ret = sum(1 for a in agents if a.is_retrofitted)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Retrofitted: {n_ret}/{model.n_agents} ({100*n_ret/model.n_agents:.1f}%)")
    print(f"Mean belief P(H1): {np.mean([a.belief for a in agents]):.4f}")
    print(f"Mean floods: {np.mean([a.flood_count for a in agents]):.2f}")
    print(f"Edges: {model.G.number_of_edges()}, "
          f"Mean degree: {np.mean([d for _, d in model.G.degree()]):.1f}")
    print(f"Neighborhoods: {model.n_neighborhoods}, "
          f"Isolated: {np.sum(model.neighborhood_labels == -1)}")
    sims = [d["similarity"] for _, _, d in model.G.edges(data=True)]
    if sims:
        print(f"Similarities: [{min(sims):.3f}, {np.mean(sims):.3f}, {max(sims):.3f}]")

    print(f"\nExporting...")
    model.get_agent_dataframe().to_csv(os.path.join(output_dir, "agent_data.csv"))
    model.get_model_dataframe().to_csv(os.path.join(output_dir, "model_data.csv"))

    print("Generating visualizations...")
    plot_adoption_curve(model, output_dir)
    plot_adoption_by_elevation(model, output_dir)
    plot_flood_history(model, output_dir)
    plot_network(model, output_dir)
    plot_comparison_observed(model, output_dir)
    plot_spatial_retrofit_map(model, output_dir)
    plot_similarity_distribution(model, output_dir)
    plot_belief_evolution(model, output_dir)
    print("Creating animation...")
    create_animation(model, output_dir)

    print(f"\n{'='*70}\nAll outputs in '{output_dir}/'\n{'='*70}")
