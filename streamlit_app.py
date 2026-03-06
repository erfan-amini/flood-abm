"""
Flood Adaptation ABM v12 — Interactive Dashboard

Two modes:
  - Research Mode: abstract/synthetic settings with noise toggles
  - Case Study Mode: upload location data and flood scenarios

References: see User_Manual_v12.docx
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import sys
import os
import io

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flood Adaptation ABM v12",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import model modules (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

import AAA_model_v12 as M
from FFF_flood import FloodGenerator
from FFF_spatial import SpatialGenerator
from FFF_attributes import AttributeGenerator
from FFF_network import NetworkBuilder
from FFF_neighborhood import identify_neighborhoods

# ---------------------------------------------------------------------------
# Matplotlib style (Palatino Linotype with fallbacks)
# ---------------------------------------------------------------------------
FONT_SIZE = 12
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "STIXGeneral", "Palatino",
                    "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2,
    "legend.fontsize": FONT_SIZE - 2,
})


# ============================================================================
# SIDEBAR — MODE SELECTION
# ============================================================================

st.sidebar.title("Flood Adaptation ABM v12")
st.sidebar.markdown("Bayesian Belief Updating")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    ["Research Mode", "Case Study Mode"],
    help=("**Research Mode**: Abstract simulation with synthetic spatial "
          "layouts and GEV flood generation. Noise toggles available.\n\n"
          "**Case Study Mode**: Upload location data (CSV) and flood "
          "scenarios for a specific site."),
)

st.sidebar.divider()

# ============================================================================
# SIDEBAR — SETTINGS
# ============================================================================

# --- General Settings ---
st.sidebar.header("General Settings")

time_steps = st.sidebar.slider("Time Steps", 10, 500, 100, step=10)
random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
n_agents = st.sidebar.slider("Number of Agents", 20, 1000, 200, step=10)

# --- Bayesian Parameters ---
st.sidebar.header("Bayesian Updating")

initial_belief = st.sidebar.slider(
    "Initial Belief P(H1)", 0.01, 0.50, 0.05, 0.01,
    help="Prior probability that situation warrants retrofitting.")
lambda_flood = st.sidebar.slider(
    "λ Flood (Ch.1)", 1.0, 3.0, 1.20, 0.01,
    help="Bayes factor per flood event.")
lambda_social = st.sidebar.slider(
    "λ Social (Ch.2 — proximity)", 1.0, 5.0, 1.50, 0.01,
    help="Bayes factor per connected neighbor retrofit.")
lambda_similarity = st.sidebar.slider(
    "λ Similarity (Ch.3 — homophily)", 1.0, 10.0, 3.00, 0.1,
    help="Bayes factor at full Jaccard similarity (S=1).")

# --- PMT Threshold ---
st.sidebar.header("PMT Threshold (Rogers, 1975)")

pmt_mean = st.sidebar.slider(
    "Threshold Mean", 0.10, 0.90, 0.50, 0.01)

# Research mode: noise toggle for PMT
if mode == "Research Mode":
    pmt_noise_on = st.sidebar.checkbox("PMT Threshold Heterogeneity", value=False,
        help="If on, thresholds are drawn from a truncated Normal. "
             "If off, all agents share the same threshold.")
    if pmt_noise_on:
        pmt_std = st.sidebar.slider("Threshold Std Dev", 0.01, 0.20, 0.10, 0.01)
        pmt_low = st.sidebar.slider("Threshold Lower Bound", 0.10, pmt_mean, 0.45, 0.01)
        pmt_high = st.sidebar.slider("Threshold Upper Bound", pmt_mean, 0.95, 0.80, 0.01)
    else:
        pmt_std = 0.0
        pmt_low = pmt_mean
        pmt_high = pmt_mean
else:
    pmt_std = st.sidebar.slider("Threshold Std Dev", 0.00, 0.20, 0.00, 0.01)
    pmt_low = st.sidebar.slider("Threshold Lower Bound", 0.10, 0.90, 0.50, 0.01)
    pmt_high = st.sidebar.slider("Threshold Upper Bound", 0.10, 0.95, 0.50, 0.01)

# ============================================================================
# ADVANCED SETTINGS (expander in sidebar)
# ============================================================================

with st.sidebar.expander("Advanced Settings"):

    st.subheader("Spatial Layout")

    if mode == "Research Mode":
        spatial_mode = st.selectbox(
            "Layout Mode",
            [("Grid with Connectors", 2), ("Random Positions", 1)],
            format_func=lambda x: x[0],
        )[1]
        grid_rows = st.number_input("Grid Rows", 1, 10, 3)
        grid_cols = st.number_input("Grid Cols", 1, 10, 4)
        n_connectors = st.number_input("Connectors", 0, 10, 2)
        slope = st.slider("Elevation Slope", 0.0, 2.0, 0.20, 0.01)

        elev_noise_on = st.checkbox("Elevation Noise", value=True,
            help="Add random noise to the elevation gradient.")
        noise_factor = 0.05 if elev_noise_on else 0.0

    else:  # Case Study
        spatial_mode = 0  # CSV
        grid_rows, grid_cols, n_connectors = 3, 4, 2
        slope = 1.0
        noise_factor = 0.0

    st.subheader("Agent Attributes")
    enable_het = st.checkbox("Enable Attribute Heterogeneity", value=True)
    n_attributes = st.number_input("Attributes per Agent", 1, 10, 2)
    n_classes = st.number_input("Classes per Attribute", 1, 10, 3)

    st.subheader("Network")
    dist_threshold = st.slider("Distance Threshold", 0.01, 0.30, 0.09, 0.01)
    dbscan_min = st.number_input("DBSCAN Min Samples", 2, 10, 4)

    st.subheader("Flood (GEV)")
    if mode == "Research Mode":
        st.caption("Return periods and flood levels for GEV fitting (Coles, 2001)")
        rp_str = st.text_input("Return Periods", "10, 20, 50, 100")
        fl_str = st.text_input("Flood Levels", "0.05, 0.10, 0.15, 0.30")
        return_periods = [int(x.strip()) for x in rp_str.split(",")]
        flood_levels = [float(x.strip()) for x in fl_str.split(",")]
    else:
        return_periods = [10, 20, 50, 100]
        flood_levels = [0.05, 0.10, 0.15, 0.30]

    st.subheader("Observed Data (optional)")
    obs_bins_str = st.text_input("Observed Bins", "0, 1, 2-3, 4+")
    obs_rates_str = st.text_input("Observed Rates (%)", "13, 18, 27, 57")
    observed_bins = [x.strip() for x in obs_bins_str.split(",")]
    observed_rates = [int(x.strip()) for x in obs_rates_str.split(",")]


# ============================================================================
# CASE STUDY MODE — FILE UPLOAD
# ============================================================================

uploaded_csv = None
uploaded_flood = None
if mode == "Case Study Mode":
    st.sidebar.divider()
    st.sidebar.header("Upload Data")
    uploaded_csv = st.sidebar.file_uploader(
        "Location CSV (columns: x, y, z)",
        type=["csv"],
        help="CSV with columns x, y, z representing agent coordinates "
             "and elevations. Values should be in [0, 1] range.",
    )
    uploaded_flood = st.sidebar.file_uploader(
        "Flood Time Series CSV (optional)",
        type=["csv"],
        help="CSV with a single column 'flood_level' containing one "
             "flood level per time step. If not provided, GEV is used.",
    )


# ============================================================================
# MAIN AREA — TABS
# ============================================================================

tab_doc, tab_sim, tab_results = st.tabs(
    ["📖 Documentation", "▶ Simulation", "📊 Results"])


# ---------------------------------------------------------------------------
# TAB: DOCUMENTATION
# ---------------------------------------------------------------------------

with tab_doc:
    st.title("Flood Adaptation ABM v12")
    st.markdown("**Bayesian Belief Updating with Three Evidence Channels**")

    st.header("Overview")
    st.markdown("""
This agent-based model simulates household flood-adaptation decisions
using Bayesian belief updating in odds form (Jaynes, 2003; Kass & Raftery, 1995).
Agents maintain a probability P(H1) representing their belief that their
situation warrants retrofitting.

**Hypotheses:**
- H1 = "my situation warrants retrofitting"
- H0 = "my situation does not warrant retrofitting"

Evidence shifts belief by multiplying the agent's odds via Bayes factors.
When P(H1) exceeds the agent's individual PMT threshold (Rogers, 1975),
the agent retrofits permanently.
""")

    st.header("Three Evidence Channels")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Ch.1: Flood Experience")
        st.markdown("""
Each flood event multiplies the agent's odds by **λ_flood**.
Safe time steps produce no update (availability heuristic;
Tversky & Kahneman, 1974).
""")
    with col2:
        st.subheader("Ch.2: Proximity")
        st.markdown("""
Connected neighbors (within distance threshold) who retrofit
deliver the full **λ_social** factor.
Binary connections, no distance decay.
""")
    with col3:
        st.subheader("Ch.3: Similarity")
        st.markdown("""
Within DBSCAN neighborhoods (Ester et al., 1996),
Jaccard attribute similarity scales the factor:
**λ_similarity ^ S(i,j)** (Jaccard, 1912).
""")

    st.header("Update Mechanism")
    st.latex(r"\text{posterior\_odds} = \text{prior\_odds} \times \text{Bayes\_factor}")
    st.latex(r"\text{where odds} = \frac{P(H_1)}{1 - P(H_1)}, \quad P(H_1) = \frac{\text{odds}}{1 + \text{odds}}")

    st.header("Modes")
    st.markdown("""
**Research Mode** uses synthetic spatial layouts (grid or random)
and GEV-generated flood levels. Noise toggles let you turn PMT threshold
heterogeneity and elevation noise on or off.

**Case Study Mode** accepts uploaded CSV data for agent locations
and optionally a flood time series. This allows the model to be
applied to a specific geographic site.
""")

    st.header("References")
    refs = [
        "Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.",
        "Ester, M. et al. (1996). A density-based algorithm for discovering clusters. *KDD-96*, 226-231.",
        "Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37-50.",
        "Jaynes, E.T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press.",
        "Kass, R.E. & Raftery, A.E. (1995). Bayes Factors. *JASA*, 90(430), 773-795.",
        "Kazil, J. et al. (2020). Utilizing Python for Agent-Based Modeling: The Mesa Framework. Springer.",
        "McPherson, M. et al. (2001). Birds of a feather: Homophily in social networks. *Annual Review of Sociology*, 27, 415-444.",
        "Rogers, R.W. (1975). A protection motivation theory of fear appeals. *Journal of Psychology*, 91(1), 93-114.",
        "Tversky, A. & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124-1131.",
    ]
    for r in refs:
        st.markdown(f"- {r}")


# ---------------------------------------------------------------------------
# HELPER: apply user parameters to model module
# ---------------------------------------------------------------------------

def apply_params():
    """Write sidebar values into the model module before instantiation."""
    M.TIME_STEPS = time_steps
    M.RANDOM_SEED = random_seed
    M.N_AGENTS = n_agents
    M.SPATIAL_MODE = spatial_mode
    M.GRID_ROWS = grid_rows
    M.GRID_COLS = grid_cols
    M.N_CONNECTORS = n_connectors
    M.SLOPE = slope
    M.NOISE_FACTOR = noise_factor
    M.ENABLE_HETEROGENEITY = enable_het
    M.N_ATTRIBUTES = n_attributes
    M.N_CLASSES = n_classes
    M.DISTANCE_THRESHOLD = dist_threshold
    M.DBSCAN_MIN_SAMPLES = dbscan_min
    M.USER_EDGES_CSV = None
    M.CSV_PATH = None
    M.INITIAL_BELIEF = initial_belief
    M.LAMBDA_FLOOD = lambda_flood
    M.LAMBDA_SOCIAL = lambda_social
    M.LAMBDA_SIMILARITY = lambda_similarity
    M.PMT_THRESHOLD_MEAN = pmt_mean
    M.PMT_THRESHOLD_STD = pmt_std
    M.PMT_THRESHOLD_LOW = pmt_low
    M.PMT_THRESHOLD_HIGH = pmt_high
    M.RETURN_PERIODS = return_periods
    M.FLOOD_LEVELS = flood_levels
    M.OBSERVED_BINS = observed_bins
    M.OBSERVED_RATES = observed_rates
    # Reparse bins after update
    M.FLOOD_BIN_RANGES = M.parse_flood_bins(observed_bins)


# ---------------------------------------------------------------------------
# HELPER: generate all figures
# ---------------------------------------------------------------------------

def make_figures(model):
    """Generate all visualization figures and return as dict."""
    figs = {}

    # Adoption curve
    df = model.get_model_dataframe()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["pct_retrofitted"], "b-", linewidth=2)
    ax.fill_between(df.index, df["pct_retrofitted"], alpha=0.15)
    ax.set(xlabel="Time Step", ylabel="Retrofitted (%)",
           title="Retrofit Adoption Over Time",
           xlim=(1, model.time_steps), ylim=(0, 100))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["adoption"] = fig

    # Adoption by elevation
    agents = list(model.agents)
    terciles = np.percentile([a.z for a in agents], [33.33, 66.67])
    groups = {"Low": [], "Medium": [], "High": []}
    for a in agents:
        if a.z <= terciles[0]: groups["Low"].append(a)
        elif a.z <= terciles[1]: groups["Medium"].append(a)
        else: groups["High"].append(a)
    rates_e, labels_e = [], []
    for label, group in groups.items():
        if group:
            rates_e.append(100 * sum(1 for a in group if a.is_retrofitted) / len(group))
            labels_e.append(f"{label}\n(n={len(group)})")
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels_e, rates_e, color=["#d62728","#ff7f0e","#2ca02c"], edgecolor="black")
    for bar, rate in zip(bars, rates_e):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{rate:.1f}%", ha="center", va="bottom")
    ax.set(xlabel="Elevation Tercile", ylabel="Retrofit Rate (%)",
           title="Retrofit Adoption by Elevation",
           ylim=(0, max(rates_e)*1.2 if rates_e else 100))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    figs["elevation"] = fig

    # Flood history
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = range(1, len(model.flood_history) + 1)
    ax.bar(steps, model.flood_history, color="steelblue", alpha=0.7, edgecolor="black")
    m = np.mean(model.flood_history)
    ax.axhline(y=m, color="red", linestyle="--", linewidth=2, label=f"Mean = {m:.3f}")
    ax.set(xlabel="Time Step", ylabel="Flood Level",
           title="Annual Flood Levels (GEV, Coles 2001)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    figs["flood"] = fig

    # Network
    fig, ax = plt.subplots(figsize=(10, 8))
    segs = []
    for u, v in model.G.edges():
        a_u, a_v = model.agents_by_node[u], model.agents_by_node[v]
        segs.append([(a_u.x, a_u.y), (a_v.x, a_v.y)])
    if segs:
        ax.add_collection(LineCollection(segs, linewidths=0.5, colors="gray",
                                         alpha=0.3, zorder=1))
    adopted = [a for a in agents if a.is_retrofitted]
    not_adopted = [a for a in agents if not a.is_retrofitted]
    if not_adopted:
        ax.scatter([a.x for a in not_adopted], [a.y for a in not_adopted],
                   c="lightgray", s=150, edgecolor="black", linewidth=0.8, zorder=2)
    if adopted:
        sc = ax.scatter([a.x for a in adopted], [a.y for a in adopted],
            c=[a.retrofit_step for a in adopted], cmap="YlGn", s=150,
            edgecolor="black", linewidth=0.8, zorder=2, vmin=1, vmax=model.current_step)
        plt.colorbar(sc, ax=ax, shrink=0.6).set_label("Retrofit Step")
    ax.set(xlabel="x", ylabel="y",
           title="Social Network (binary connections)",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Patch(facecolor="lightgray", edgecolor="black", label="Not Retrofitted"),
        Patch(facecolor="#31a354", edgecolor="black", label="Retrofitted")],
        loc="upper right")
    plt.tight_layout()
    figs["network"] = fig

    # Comparison with observed
    bins_dict = {b: {"total": 0, "retrofitted": 0} for b in observed_bins}
    for a in agents:
        cat = M.categorize_flood_count(a.flood_count)
        bins_dict[cat]["total"] += 1
        if a.is_retrofitted: bins_dict[cat]["retrofitted"] += 1
    model_rates = [100*bins_dict[b]["retrofitted"]/bins_dict[b]["total"]
                   if bins_dict[b]["total"]>0 else 0 for b in observed_bins]
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(observed_bins)); width = 0.35
    ax.bar(x_pos-width/2, model_rates, width, label="Model",
           color="steelblue", edgecolor="black")
    ax.bar(x_pos+width/2, observed_rates, width, label="Observed",
           color="coral", edgecolor="black")
    ax.set(xlabel="Flood Count Category", ylabel="Retrofit Rate (%)",
           title="Model vs Observed Retrofit Rates")
    ax.set_xticks(x_pos); ax.set_xticklabels(observed_bins)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    figs["comparison"] = fig

    # Similarity distribution
    sims = [d["similarity"] for _, _, d in model.G.edges(data=True)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sims, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(sims), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(sims):.3f}")
    ax.set(xlabel="Jaccard Similarity S(i,j)", ylabel="Frequency",
           title="Distribution of Attribute Similarity (Jaccard, 1912)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    figs["similarity"] = fig

    # Belief evolution
    df_model = model.get_model_dataframe()
    df_agent = model.get_agent_dataframe()
    bsteps = df_model.index.values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    p10, p90 = [], []
    for s in bsteps:
        sb = df_agent.xs(s, level="Step")["belief"].values
        p10.append(np.percentile(sb, 10))
        p90.append(np.percentile(sb, 90))
    ax1.fill_between(bsteps, p10, p90, alpha=0.2, color="blue",
                     label="10th\u201390th percentile")
    ax1.plot(bsteps, df_model["mean_belief"], "b-", linewidth=2,
             label="Mean $P(H_1)$")
    final_data = df_agent.xs(bsteps[-1], level="Step")
    thr_vals = final_data["pmt_threshold"].values
    thr_mean = thr_vals.mean()
    if pmt_std > 0:
        tp10, tp90 = np.percentile(thr_vals, [10, 90])
        ax1.axhspan(tp10, tp90, color="red", alpha=0.08,
                    label=f"Threshold 10th\u201390th: [{tp10:.2f}, {tp90:.2f}]")
    ax1.axhline(y=thr_mean, color="red", linestyle="--",
                label=f"Mean threshold = {thr_mean:.2f}")
    ax1.axhline(y=initial_belief, color="gray", linestyle=":",
                label=f"Prior = {initial_belief}")
    ax1.set(xlabel="Time Step", ylabel="$P(H_1)$",
            title="Bayesian Belief Evolution")
    ax1.legend(fontsize=FONT_SIZE-3); ax1.grid(True, alpha=0.3)
    final_beliefs = final_data["belief"].values
    ax2.hist(final_beliefs, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax2.axvline(x=thr_mean, color="red", linestyle="--", linewidth=2,
                label=f"Mean threshold = {thr_mean:.2f}")
    ax2.set(xlabel="$P(H_1)$", ylabel="Number of Agents",
            title=f"Final Belief Distribution (Step {bsteps[-1]})")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    figs["belief"] = fig

    return figs


# ---------------------------------------------------------------------------
# TAB: SIMULATION
# ---------------------------------------------------------------------------

with tab_sim:
    st.title("Run Simulation")

    # Summarize current settings
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Agents", n_agents)
        st.metric("Time Steps", time_steps)
    with c2:
        st.metric("λ Flood", f"{lambda_flood:.2f}")
        st.metric("λ Social", f"{lambda_social:.2f}")
    with c3:
        st.metric("λ Similarity", f"{lambda_similarity:.2f}")
        st.metric("PMT Threshold", f"{pmt_mean:.2f}")

    st.divider()

    # Validation for Case Study
    if mode == "Case Study Mode" and uploaded_csv is None:
        st.warning("Upload a location CSV in the sidebar to run Case Study Mode.")

    # Run button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            apply_params()

            # Handle Case Study CSV
            if mode == "Case Study Mode" and uploaded_csv is not None:
                uploaded_csv.seek(0)
                csv_df = pd.read_csv(uploaded_csv)
                csv_path = "/tmp/_abm_upload.csv"
                csv_df.to_csv(csv_path, index=False)
                M.N_AGENTS = len(csv_df)
                M.SPATIAL_MODE = 0
                M.CSV_PATH = csv_path

            model = M.FloodAdaptationModel()

            # Inject uploaded flood time series if provided
            if (mode == "Case Study Mode"
                    and uploaded_flood is not None):
                uploaded_flood.seek(0)
                flood_ts = pd.read_csv(uploaded_flood)["flood_level"].values
                class FixedFlood:
                    def __init__(self, levels):
                        self._levels = levels; self._idx = 0
                    def sample(self):
                        if self._idx < len(self._levels):
                            v = self._levels[self._idx]; self._idx += 1
                            return float(np.clip(v, 0, 1))
                        return 0.0
                model.flood_generator = FixedFlood(flood_ts)

            model.run()

        # Store in session state
        st.session_state["model"] = model
        st.session_state["figs"] = make_figures(model)

        # Summary
        agents = list(model.agents)
        n_ret = sum(1 for a in agents if a.is_retrofitted)

        st.success(f"Simulation complete. "
                   f"Retrofitted: {n_ret}/{model.n_agents} "
                   f"({100*n_ret/model.n_agents:.1f}%)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrofitted", f"{n_ret}/{model.n_agents}")
        c2.metric("Mean Belief", f"{np.mean([a.belief for a in agents]):.4f}")
        c3.metric("Mean Floods", f"{np.mean([a.flood_count for a in agents]):.1f}")
        c4.metric("Neighborhoods", f"{model.n_neighborhoods}")

        st.info("Go to the **Results** tab to view plots.")


# ---------------------------------------------------------------------------
# TAB: RESULTS
# ---------------------------------------------------------------------------

with tab_results:
    st.title("Results")

    if "figs" not in st.session_state:
        st.info("Run a simulation first to see results here.")
    else:
        figs = st.session_state["figs"]
        model = st.session_state["model"]

        # Adoption curve
        st.header("Adoption Dynamics")
        st.pyplot(figs["adoption"])

        # Two-column: elevation + comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Adoption by Elevation")
            st.pyplot(figs["elevation"])
        with col2:
            st.subheader("Model vs Observed")
            st.pyplot(figs["comparison"])

        # Belief evolution
        st.header("Belief Evolution")
        st.pyplot(figs["belief"])

        # Flood history
        st.header("Flood History")
        st.pyplot(figs["flood"])

        # Two-column: network + similarity
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Social Network")
            st.pyplot(figs["network"])
        with col2:
            st.subheader("Similarity Distribution")
            st.pyplot(figs["similarity"])

        # Download data
        st.header("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            agent_csv = model.get_agent_dataframe().to_csv()
            st.download_button("Download Agent Data (CSV)",
                               agent_csv, "agent_data.csv", "text/csv")
        with col2:
            model_csv = model.get_model_dataframe().to_csv()
            st.download_button("Download Model Data (CSV)",
                               model_csv, "model_data.csv", "text/csv")
