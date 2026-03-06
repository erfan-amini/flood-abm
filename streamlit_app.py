"""
Flood Adaptation ABM — Interactive Dashboard

Two modes:
  - Research Mode: abstract/synthetic settings with noise toggles
  - Case Study Mode: upload location data and flood scenarios
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker
import sys
import os
import time

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flood Adaptation ABM",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0F4F8 0%, #E2ECF5 100%);
}
[data-testid="stSidebar"] h1 {
    color: #2E75B6;
    font-size: 1.5rem;
}
[data-testid="stSidebar"] h2 {
    color: #1A5A96;
    border-bottom: 2px solid #0EA5E9;
    padding-bottom: 0.3rem;
    font-size: 1.1rem;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-size: 1.05rem;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: #2E75B6 !important;
    color: #2E75B6 !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F8FBFF, #EBF3FB);
    border: 1px solid #D0DEF0;
    border-left: 4px solid #2E75B6;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetric"] label {
    color: #1A5A96 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #2E75B6 !important;
    font-weight: 700;
}

/* ---- Headers ---- */
.main h1 { color: #2E75B6; }
.main h2 { color: #1A5A96; border-bottom: 1px solid #D0DEF0; padding-bottom: 0.3rem; }
.main h3 { color: #0E6BA8; }

/* ---- Primary button ---- */
button[kind="primary"], .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2E75B6, #0EA5E9) !important;
    border: none !important;
    font-weight: 600;
    color: white !important;
}

/* ---- Dividers ---- */
hr { border-color: #D0DEF0 !important; }

/* ---- Alert boxes ---- */
[data-testid="stAlert"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Import model modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
import AAA_model_v12 as M
from FFF_neighborhood import identify_neighborhoods

# ---------------------------------------------------------------------------
# Matplotlib — LaTeX-style configuration
# ---------------------------------------------------------------------------
FS = 11
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Palatino Linotype", "Palatino",
                    "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "font.size": FS,
    "axes.titlesize": FS + 1,
    "axes.labelsize": FS,
    "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 2,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
})


# ============================================================================
# SIDEBAR — MODE + GENERAL SETTINGS ONLY
# ============================================================================

st.sidebar.title("Flood Adaptation ABM")
st.sidebar.caption("Bayesian Belief Updating Model")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    ["Research Mode", "Case Study Mode"],
    help=("**Research Mode**: Synthetic spatial layouts, GEV floods, "
          "noise toggles.\n\n"
          "**Case Study Mode**: Upload location CSV and flood data."),
)

st.sidebar.divider()
st.sidebar.header("General Settings")

time_steps = st.sidebar.slider("Time Steps", 10, 500, 100, step=10)
random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
n_agents = st.sidebar.slider("Number of Agents", 20, 1000, 200, step=10)

st.sidebar.divider()
st.sidebar.header("Bayesian Updating")

initial_belief = st.sidebar.slider(
    "Initial Belief $P(H_1)$", 0.01, 0.50, 0.05, 0.01)
st.sidebar.markdown("**Channel 1: Personal Flood Experience**")
lambda_flood = st.sidebar.slider(
    "$\\lambda_{\\mathrm{flood}}$", 1.0, 3.0, 1.20, 0.01,
    help="Bayes factor applied when an agent is personally flooded.")

st.sidebar.markdown("**Channel 2: Proximity-Based Social Learning**")
lambda_social = st.sidebar.slider(
    "$\\lambda_{\\mathrm{social}}$", 1.0, 5.0, 1.50, 0.01,
    help="Bayes factor applied when a connected neighbor retrofits.")

st.sidebar.markdown("**Channel 3: Similarity-Based Social Learning**")
lambda_similarity = st.sidebar.slider(
    "$\\lambda_{\\mathrm{similarity}}$", 1.0, 10.0, 3.00, 0.1,
    help="Base Bayes factor scaled by Jaccard similarity within neighborhoods.")

st.sidebar.divider()
st.sidebar.header("PMT Threshold")

pmt_mean = st.sidebar.slider("Threshold Mean", 0.10, 0.90, 0.50, 0.01)

if mode == "Research Mode":
    pmt_noise_on = st.sidebar.checkbox(
        "Threshold Heterogeneity", value=False,
        help="Draw individual thresholds from a truncated Normal.")
    if pmt_noise_on:
        pmt_std = st.sidebar.slider("Std Dev", 0.01, 0.20, 0.10, 0.01)
        pmt_low = st.sidebar.slider("Lower Bound", 0.10, pmt_mean, 0.45, 0.01)
        pmt_high = st.sidebar.slider("Upper Bound", pmt_mean, 0.95, 0.80, 0.01)
    else:
        pmt_std, pmt_low, pmt_high = 0.0, pmt_mean, pmt_mean
else:
    pmt_std = st.sidebar.slider("Std Dev", 0.00, 0.20, 0.00, 0.01)
    pmt_low = st.sidebar.slider("Lower Bound", 0.10, 0.90, 0.50, 0.01)
    pmt_high = st.sidebar.slider("Upper Bound", 0.10, 0.95, 0.50, 0.01)

# --- Case Study file uploads ---
uploaded_csv = None
uploaded_flood = None
if mode == "Case Study Mode":
    st.sidebar.divider()
    st.sidebar.header("Upload Data")
    uploaded_csv = st.sidebar.file_uploader(
        "Location CSV (columns: x, y, z)", type=["csv"])
    uploaded_flood = st.sidebar.file_uploader(
        "Flood Time Series CSV (optional)", type=["csv"],
        help="Column `flood_level`, one value per time step.")


# ============================================================================
# MAIN TABS
# ============================================================================

# ---------------------------------------------------------------------------
# Main title and subtitle
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 1.2rem 0 0.6rem 0;">
    <h1 style="margin:0; font-size:2.4rem; color:#2E75B6; border:none;">
        Flood Adaptation Agent-Based Model
    </h1>
    <p style="margin:0.3rem 0 0 0; font-size:1.15rem; color:#555; font-style:italic;">
        Bayesian Belief Updating with Three Evidence Channels
    </p>
    <hr style="margin:0.8rem auto 0 auto; width:60%; border:none;
               height:3px; background: linear-gradient(90deg, #2E75B6, #0EA5E9, #10B981);">
</div>
""", unsafe_allow_html=True)

tab_doc, tab_settings, tab_sim, tab_results = st.tabs(
    ["Documentation", "Advanced Settings", "Run Simulation", "Results"])


# ============================================================================
# TAB: DOCUMENTATION
# ============================================================================

with tab_doc:
    st.header("1. Purpose")
    st.markdown("""
This model simulates how households in flood-prone areas decide whether
to retrofit their properties. Each household (agent) holds a subjective
belief about whether its situation warrants protective action. That belief
is updated over time as the agent experiences floods and observes the
behavior of its neighbors. When the belief crosses a decision threshold,
the agent retrofits permanently.

The model addresses a central question in natural hazard adaptation:
why do some households act quickly while others delay for years, even
when facing the same objective risk?
""")

    st.header("2. Hypotheses")
    st.markdown("""
Each agent evaluates two competing assessments of its own circumstances:

| Hypothesis | Meaning |
|:---:|---|
| $H_1$ | "My situation warrants retrofitting" |
| $H_0$ | "My situation does not warrant retrofitting" |

The agent maintains $P(H_1)$, updated through evidence. These are
subjective assessments, not statements about geophysical parameters.
The decision to retrofit is separate from the inference process.
""")

    st.header("3. Bayesian Update Mechanism")
    st.markdown("""
Belief is updated using Bayes' theorem in odds form
(Jaynes, 2003, Ch. 4; Kass & Raftery, 1995):
""")
    st.latex(r"""
\text{posterior odds} = \text{prior odds} \times
\underbrace{\frac{P(\text{evidence} \mid H_1)}
{P(\text{evidence} \mid H_0)}}_{\text{Bayes factor}}
""")
    st.markdown("""
where $\\text{odds} = P(H_1) \\,/\\, (1 - P(H_1))$. After multiplying,
the agent converts back: $P(H_1) = \\text{odds} \\,/\\, (1 + \\text{odds})$.

The Bayes factor measures how much more the observed evidence is expected
under $H_1$ than under $H_0$. A factor greater than 1 shifts belief
toward $H_1$. A factor of 1 is uninformative. This form is algebraically
equivalent to the standard formulation of Bayes' theorem.
""")

    st.header("4. Three Evidence Channels")

    st.markdown("""
| Channel | Parameter | Mechanism | Scope |
|:---:|:---:|---|---|
| 1 | $\\lambda_{\\mathrm{flood}}$ | Personal flood experience multiplies odds | Individual agent |
| 2 | $\\lambda_{\\mathrm{social}}$ | Observing a connected neighbor retrofit | Binary network edges |
| 3 | $\\lambda_{\\mathrm{similarity}}$ | Similarity-scaled learning within neighborhoods | DBSCAN clusters |
""")

    st.subheader("Channel 1: Personal Flood Experience")
    st.markdown("""
Each time step, a global flood level is drawn from a Generalized Extreme
Value (GEV) distribution (Coles, 2001). Agents whose elevation is below
the flood level are flooded. A flood event multiplies the agent's odds by
$\\lambda_{\\mathrm{flood}}$.

Safe time steps produce no update. This asymmetry follows the availability
heuristic (Tversky & Kahneman, 1974): vivid, dramatic flood events are
cognitively processed as evidence, while uneventful periods are
psychologically inert. The result is a monotonically non-decreasing
belief trajectory from personal experience alone.
""")

    st.subheader("Channel 2: Proximity-Based Social Learning")
    st.markdown("""
Network connections are binary. Two agents are connected if their
Euclidean distance is within a threshold, and not connected otherwise.
No continuous distance decay is applied. When an agent first observes
that a connected neighbor has retrofitted, its odds are multiplied
by $\\lambda_{\\mathrm{social}}$.

This captures the direct influence of living near someone who has taken
action, regardless of demographic similarity. Each neighbor is counted
once (one-shot observation).
""")

    st.subheader("Channel 3: Similarity-Based Social Learning")
    st.markdown("""
Spatial neighborhoods are identified using DBSCAN density-based
clustering (Ester et al., 1996). Agents in the same cluster share a
neighborhood. Within a neighborhood, the Jaccard similarity $S(i,j)$
between agents' attribute vectors (Jaccard, 1912) scales the Bayes factor:
""")
    st.latex(r"\text{effective factor} = \lambda_{\mathrm{similarity}}^{\,S(i,j)}")
    st.markdown("""
When $S = 1$ (identical attributes), the full factor applies. When $S = 0$
(no shared attributes), the effective factor is 1 (no update). This
captures homophily: agents are more influenced by those who share
their characteristics (McPherson et al., 2001).

Agents in different neighborhoods, or connector agents labeled as noise
by DBSCAN, do not participate in this channel.
""")

    st.header("5. Decision Rule")
    st.markdown("""
When $P(H_1) \\geq$ the agent's individual threshold, the agent
retrofits permanently. Thresholds can be homogeneous (all agents share
the same value) or drawn from a truncated Normal distribution to
represent variation in self-efficacy and response costs
(Rogers, 1975). Retrofit is irreversible: once an agent retrofits, it
exits the learning loop.
""")

    st.header("6. Spatial Structure")
    st.markdown("""
**Research Mode** generates agents on a grid of connected neighborhoods
with bridge (connector) agents between them, or at random positions.
Elevation follows a linear gradient ($z = \\text{slope} \\times x$)
with optional noise.

**Case Study Mode** accepts a CSV file with columns `x`, `y`, `z`
specifying each agent's position and elevation.
""")

    st.header("7. Modes of Operation")

    st.subheader("Research Mode")
    st.markdown("""
Designed for theoretical exploration. Uses synthetic layouts and
GEV-generated flood levels. Noise toggles allow turning on or off:
- **PMT threshold heterogeneity** (truncated Normal vs. homogeneous)
- **Elevation noise** (random perturbation of the elevation gradient)

This mode is useful for sensitivity analysis and understanding how
each model component contributes to aggregate adoption patterns.
""")

    st.subheader("Case Study Mode")
    st.markdown("""
Designed for application to a specific site. Upload:
- **Location CSV** (required): columns `x`, `y`, `z` with values in [0, 1].
- **Flood time series CSV** (optional): column `flood_level`, one value
  per time step. If not provided, GEV generation is used.
""")

    st.header("8. Interpreting the Bayes Factors")
    st.markdown("""
| Factor | Meaning |
|:---:|---|
| $\\lambda = 1.0$ | Evidence is uninformative; no belief change |
| $\\lambda = 1.2$ | Each event increases odds by 20% |
| $\\lambda = 2.0$ | Each event doubles the odds |
| $\\lambda = 3.0$ | Each event triples the odds |

A neighbor who is both connected and in the same neighborhood triggers
Channels 2 and 3. A neighbor who is connected but in a different
neighborhood triggers only Channel 2.
""")

    st.subheader("Worked Example")
    st.markdown("""
Consider an agent with initial belief $P(H_1) = 0.05$ and
$\\lambda_{\\mathrm{flood}} = 1.5$, $\\lambda_{\\mathrm{social}} = 2.0$,
threshold $= 0.50$.

| Step | Event | Prior Odds | Bayes Factor | Posterior Odds | $P(H_1)$ |
|:---:|---|:---:|:---:|:---:|:---:|
| 0 | Initial state | 0.0526 | -- | 0.0526 | 0.050 |
| 1 | Flooded | 0.0526 | 1.50 | 0.0789 | 0.073 |
| 2 | Safe (no update) | 0.0789 | 1.00 | 0.0789 | 0.073 |
| 3 | Flooded + neighbor retrofits | 0.0789 | 1.50 x 2.00 = 3.00 | 0.2368 | 0.191 |

The agent has not yet reached the threshold of 0.50, so it has not
retrofitted. Note how a single step with multiple evidence sources
(flood + social) produces a compound update.
""")

    st.header("9. Key Properties")
    st.markdown("""
- **Three separable channels.** Each can be disabled by setting its
  factor to 1.0.
- **Binary connections.** No distance decay. Agents are connected or not.
- **Similarity operates within neighborhoods.** DBSCAN clustering
  identifies spatial communities; homophily effects apply only within them.
- **Monotonically non-decreasing belief.** All factors $\\geq 1$;
  belief never drops.
- **Heterogeneous thresholds** create timing diversity among agents
  with identical exposure.
""")

    st.header("10. Model Architecture and Flow")
    st.markdown("""
The model proceeds in two phases: **initialization** and **stepping**.

**Initialization:**
1. **Spatial placement** -- Agents are assigned $(x, y)$ positions via grid layout,
   random placement, or uploaded CSV.
2. **Elevation assignment** -- Elevation $z$ follows a linear gradient
   ($z = \\text{slope} \\times x$) with optional Gaussian noise, or is read from CSV.
3. **Attribute assignment** -- Each agent receives a vector of categorical attributes
   (e.g., income class, housing type) used for Jaccard similarity.
4. **Network construction** -- Binary edges are created between agents within the
   Euclidean distance threshold. No distance decay is applied.
5. **Neighborhood detection** -- DBSCAN clusters agents into spatial neighborhoods.
   Noise points (connectors) are excluded from similarity-based learning.
6. **Belief initialization** -- All agents start with the same prior $P(H_1)$.
   Individual PMT thresholds are drawn (homogeneous or truncated Normal).

**Step Loop (repeated for each time step):**
1. Draw a global flood level from the GEV distribution (or uploaded series).
2. **Channel 1**: Each agent with $z <$ flood level is flooded; odds $\\times \\lambda_{\\mathrm{flood}}$.
3. **Channel 2**: Each agent checks connected neighbors for new retrofits;
   odds $\\times \\lambda_{\\mathrm{social}}$ per newly observed retrofit.
4. **Channel 3**: Within DBSCAN neighborhoods, each agent checks neighbors for
   new retrofits; effective factor $= \\lambda_{\\mathrm{similarity}}^{S(i,j)}$.
5. **Decision**: If $P(H_1) \\geq$ threshold, the agent retrofits permanently.
6. Record step-level statistics (mean belief, retrofit count, flood level).
""")

    st.header("11. Parameter Sensitivity Guide")
    st.markdown("""
| Parameter | Range | Default | Effect |
|---|:---:|:---:|---|
| $\\lambda_{\\mathrm{flood}}$ | 1.0 -- 3.0 | 1.20 | Higher values accelerate belief growth from personal flood experience. Most direct driver of low-elevation adoption. |
| $\\lambda_{\\mathrm{social}}$ | 1.0 -- 5.0 | 1.50 | Higher values amplify social contagion through network connections. Creates cascading adoption waves. |
| $\\lambda_{\\mathrm{similarity}}$ | 1.0 -- 10.0 | 3.00 | Higher values strengthen homophily-driven learning. Effect is modulated by Jaccard similarity $S(i,j)$. |
| Initial Belief $P(H_1)$ | 0.01 -- 0.50 | 0.05 | Higher prior beliefs bring agents closer to threshold from the start. |
| PMT Threshold | 0.10 -- 0.90 | 0.50 | Lower thresholds make agents retrofit sooner; higher thresholds require more evidence. |
| Threshold Std Dev | 0.00 -- 0.20 | 0.00 | Positive values create agent heterogeneity, producing gradual adoption curves rather than sharp jumps. |
| Distance Threshold | 0.01 -- 0.30 | 0.09 | Larger values create denser networks with more connections, amplifying Channel 2. |
| DBSCAN Min Samples | 2 -- 10 | 4 | Lower values create more (smaller) neighborhoods; higher values require denser clusters. |
| Elevation Slope | 0.0 -- 2.0 | 0.20 | Steeper slopes create greater elevation variation, differentiating flood exposure across agents. |
| Number of Agents | 20 -- 1000 | 200 | More agents provide smoother statistics but increase computation time. |

**Recommended exploration order:** Start by varying $\\lambda_{\\mathrm{flood}}$ and the PMT
threshold to understand individual-level dynamics. Then adjust
$\\lambda_{\\mathrm{social}}$ and the distance threshold to explore social contagion.
Finally, vary $\\lambda_{\\mathrm{similarity}}$, attributes, and DBSCAN settings
to investigate homophily effects.
""")

    st.header("12. Output Interpretation Guide")
    st.markdown("""
**Figure 1 -- Adoption and Flood History:**
- *Panel (a)*: The retrofit adoption curve shows the cumulative percentage of
  agents who have retrofitted over time. An S-shaped curve suggests social
  contagion amplifying initial flood-driven adoption. A step-function pattern
  indicates adoption triggered by specific large flood events. A slow linear
  rise suggests weak evidence accumulation.
- *Panel (b)*: Flood levels drawn from the GEV distribution. The red dashed line
  shows the mean. Occasional spikes above the mean represent extreme events
  that flood many agents simultaneously.

**Figure 2 -- Elevation and Observed Comparison:**
- *Panel (a)*: Adoption rates by elevation tercile. Low-elevation agents should
  retrofit more frequently because they are flooded more often. If medium or
  high terciles show comparable rates, social learning is dominant.
- *Panel (b)*: Model predictions vs. observed data (if provided). Alignment
  indicates good calibration; systematic over- or under-prediction suggests
  parameter adjustment is needed.

**Figure 3 -- Belief Evolution:**
- *Panel (a)*: Mean belief trajectory with the 10th--90th percentile band.
  The red dashed line marks the decision threshold. When the mean belief
  crosses the threshold, approximately half the agents have retrofitted.
- *Panel (b)*: Histogram of final beliefs. A bimodal distribution (peaks near 0
  and near 1) indicates clear separation between convinced and unconvinced agents.

**Figure 4 -- Network and Similarity:**
- *Panel (a)*: Social network layout. Node color indicates retrofit status
  (gray = not retrofitted, green shades = retrofitted, colored by timing).
  Numbers inside nodes show each agent's personal flood count. Clusters of
  green nodes indicate social contagion effects.
- *Panel (b)*: Distribution of Jaccard similarities across all network edges.
  Higher mean similarity amplifies Channel 3 effects.

**Figure 5 -- Spatial Map:**
- Elevation-colored scatter plot with green overlay for retrofitted agents.
  Look for spatial clustering of adoption in low-elevation zones and along
  network-connected corridors.
""")

    with st.expander("13. Glossary of Terms"):
        st.markdown("""
| Term | Definition |
|---|---|
| **Agent** | A simulated household that holds a belief, makes decisions, and interacts with neighbors. |
| **Availability Heuristic** | Cognitive bias where vivid events (floods) are weighted more heavily than non-events (safe years). |
| **Bayes Factor** | The likelihood ratio $P(\\text{evidence} \\mid H_1) / P(\\text{evidence} \\mid H_0)$; measures evidential strength. |
| **Belief** | The agent's subjective probability $P(H_1)$ that its situation warrants retrofitting. |
| **Channel** | One of three independent evidence pathways that update an agent's belief. |
| **Connector Agent** | A bridge agent placed between grid neighborhoods; classified as noise by DBSCAN. |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise; groups nearby agents into neighborhoods. |
| **GEV Distribution** | Generalized Extreme Value distribution; models the probability of rare flood events. |
| **Homophily** | The tendency for similar individuals to influence each other more strongly. |
| **Jaccard Similarity** | $S(i,j) = |A_i \\cap A_j| / |A_i \\cup A_j|$; measures attribute overlap between two agents. |
| **Neighborhood** | A spatial cluster identified by DBSCAN within which similarity-based learning operates. |
| **Odds Form** | $\\text{odds} = P(H_1) / (1 - P(H_1))$; multiplicative updates are natural in this form. |
| **PMT Threshold** | Protection Motivation Theory threshold; the belief level at which an agent decides to retrofit. |
| **Posterior** | The updated belief after incorporating new evidence. |
| **Prior** | The belief before incorporating new evidence. |
| **Retrofit** | A permanent protective action taken by an agent once its belief crosses the threshold. |
| **Return Period** | The average time between flood events of a given magnitude (e.g., 100-year flood). |
""")

    with st.expander("14. References"):
        st.markdown("""
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*.
  Springer Series in Statistics. Springer-Verlag London.

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based
  algorithm for discovering clusters in large spatial databases with noise.
  In *Proceedings of the 2nd International Conference on Knowledge Discovery
  and Data Mining (KDD-96)*, pp. 226--231. AAAI Press.

- Jaccard, P. (1912). The distribution of the flora in the alpine zone.
  *New Phytologist*, 11(2), 37--50.

- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*.
  Cambridge University Press.

- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the
  American Statistical Association*, 90(430), 773--795.

- Kazil, J., Masad, D., & Crooks, A. (2020). Utilizing Python for
  agent-based modeling: The Mesa framework. In *Social, Cultural, and
  Behavioral Modeling (SBP-BRiMS 2020)*, pp. 308--317. Springer.

- McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a feather:
  Homophily in social networks. *Annual Review of Sociology*, 27, 415--444.

- Rogers, R. W. (1975). A protection motivation theory of fear appeals and
  attitude change. *Journal of Psychology*, 91(1), 93--114.

- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics
  and biases. *Science*, 185(4157), 1124--1131.
""")


# ============================================================================
# TAB: ADVANCED SETTINGS
# ============================================================================

with tab_settings:
    st.title("Advanced Settings")
    st.markdown("Adjust spatial, network, attribute, and flood parameters below.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Spatial Layout")
        if mode == "Research Mode":
            spatial_mode = st.selectbox(
                "Layout Mode",
                options=[2, 1],
                format_func=lambda x: {2: "Grid with Connectors",
                                        1: "Random Positions"}[x],
            )
            grid_rows = st.number_input("Grid Rows", 1, 10, 3)
            grid_cols = st.number_input("Grid Cols", 1, 10, 4)
            n_connectors = st.number_input("Connectors Between Neighborhoods", 0, 10, 2)
            slope = st.slider("Elevation Slope", 0.0, 2.0, 0.20, 0.01)
            elev_noise_on = st.checkbox("Elevation Noise", value=True,
                help="Add random perturbation to elevation gradient.")
            noise_factor = 0.05 if elev_noise_on else 0.0
        else:
            spatial_mode = 0
            grid_rows, grid_cols, n_connectors = 3, 4, 2
            slope, noise_factor = 1.0, 0.0
            st.info("Spatial layout is determined by uploaded CSV in Case Study Mode.")

        st.subheader("Agent Attributes")
        enable_het = st.checkbox("Enable Attribute Heterogeneity", value=True)
        n_attributes = st.number_input("Attributes per Agent", 1, 10, 2)
        n_classes = st.number_input("Classes per Attribute", 1, 10, 3)

    with col_right:
        st.subheader("Network")
        dist_threshold = st.slider("Distance Threshold", 0.01, 0.30, 0.09, 0.01,
            help="Max Euclidean distance for a binary connection.")
        dbscan_min = st.number_input("DBSCAN Min Samples", 2, 10, 4,
            help="Minimum agents (incl. self) to form a neighborhood core.")

        st.subheader("Flood Generation (GEV)")
        if mode == "Research Mode":
            rp_str = st.text_input("Return Periods", "10, 20, 50, 100")
            fl_str = st.text_input("Flood Levels", "0.05, 0.10, 0.15, 0.30")
            return_periods = [int(x.strip()) for x in rp_str.split(",")]
            flood_levels = [float(x.strip()) for x in fl_str.split(",")]
        else:
            return_periods = [10, 20, 50, 100]
            flood_levels = [0.05, 0.10, 0.15, 0.30]
            st.info("Flood data from uploaded CSV or GEV defaults in Case Study Mode.")

        st.subheader("Observed Data (optional)")
        obs_bins_str = st.text_input("Observed Bins", "0, 1, 2-3, 4+")
        obs_rates_str = st.text_input("Observed Rates (%)", "13, 18, 27, 57")
        observed_bins = [x.strip() for x in obs_bins_str.split(",")]
        observed_rates = [int(x.strip()) for x in obs_rates_str.split(",")]


# ============================================================================
# HELPER: apply parameters
# ============================================================================

def apply_params():
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
    M.FLOOD_BIN_RANGES = M.parse_flood_bins(observed_bins)


# ============================================================================
# HELPER: generate figures
# ============================================================================

def make_figures(model):
    agents = list(model.agents)
    df_model = model.get_model_dataframe()
    df_agent = model.get_agent_dataframe()
    steps = df_model.index.values
    figs = {}

    # --- Figure 1: Adoption Curve + Flood History (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(df_model.index, df_model["pct_retrofitted"], "b-", linewidth=1.8)
    ax1.fill_between(df_model.index, df_model["pct_retrofitted"], alpha=0.12)
    ax1.set(xlabel="Time Step", ylabel="Retrofitted (%)",
            title="(a) Retrofit Adoption Over Time",
            xlim=(1, model.time_steps), ylim=(0, 100))

    ax2.bar(range(1, len(model.flood_history)+1), model.flood_history,
            color="steelblue", alpha=0.7, edgecolor="none", linewidth=0)
    m = np.mean(model.flood_history)
    ax2.axhline(y=m, color="red", linestyle="--", linewidth=1.2,
                label=f"Mean = {m:.3f}")
    ax2.set(xlabel="Time Step", ylabel="Flood Level",
            title="(b) Flood Levels (GEV)")
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["adoption_flood"] = fig

    # --- Figure 2: Elevation + Comparison (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Elevation terciles
    terciles = np.percentile([a.z for a in agents], [33.33, 66.67])
    groups = {"Low": [], "Medium": [], "High": []}
    for a in agents:
        if a.z <= terciles[0]: groups["Low"].append(a)
        elif a.z <= terciles[1]: groups["Medium"].append(a)
        else: groups["High"].append(a)
    rates_e, labels_e = [], []
    for label, group in groups.items():
        if group:
            rates_e.append(100*sum(1 for a in group if a.is_retrofitted)/len(group))
            labels_e.append(f"{label}\n($n$={len(group)})")
    bars = ax1.bar(labels_e, rates_e,
                   color=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black",
                   linewidth=0.6)
    for bar, rate in zip(bars, rates_e):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=FS-1)
    ax1.set(xlabel="Elevation Tercile", ylabel="Retrofit Rate (%)",
            title="(a) Adoption by Elevation",
            ylim=(0, max(rates_e)*1.25 if rates_e else 100))

    # Model vs Observed
    bins_dict = {b: {"total": 0, "retrofitted": 0} for b in observed_bins}
    for a in agents:
        cat = M.categorize_flood_count(a.flood_count)
        bins_dict[cat]["total"] += 1
        if a.is_retrofitted: bins_dict[cat]["retrofitted"] += 1
    model_rates = [100*bins_dict[b]["retrofitted"]/bins_dict[b]["total"]
                   if bins_dict[b]["total"] > 0 else 0 for b in observed_bins]
    x_pos = np.arange(len(observed_bins)); width = 0.35
    ax2.bar(x_pos-width/2, model_rates, width, label="Model",
            color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos+width/2, observed_rates, width, label="Observed",
            color="coral", edgecolor="black", linewidth=0.5)
    for i, (mr, orr) in enumerate(zip(model_rates, observed_rates)):
        if mr > 0:
            ax2.text(i-width/2, mr+1, f"{mr:.0f}%", ha="center",
                     fontsize=FS-2, color="steelblue")
        if orr > 0:
            ax2.text(i+width/2, orr+1, f"{orr}%", ha="center",
                     fontsize=FS-2, color="coral")
    ax2.set(xlabel="Flood Count Category", ylabel="Retrofit Rate (%)",
            title="(b) Model vs Observed")
    ax2.set_xticks(x_pos); ax2.set_xticklabels(observed_bins)
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["elev_comparison"] = fig

    # --- Figure 3: Belief Evolution (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    p10, p90 = [], []
    for s in steps:
        sb = df_agent.xs(s, level="Step")["belief"].values
        p10.append(np.percentile(sb, 10))
        p90.append(np.percentile(sb, 90))
    ax1.fill_between(steps, p10, p90, alpha=0.15, color="blue",
                     label="10th\u201390th pctl.")
    ax1.plot(steps, df_model["mean_belief"], "b-", linewidth=1.8,
             label="Mean $P(H_1)$")
    final_data = df_agent.xs(steps[-1], level="Step")
    thr_vals = final_data["pmt_threshold"].values
    thr_mean = thr_vals.mean()
    if pmt_std > 0:
        tp10, tp90 = np.percentile(thr_vals, [10, 90])
        ax1.axhspan(tp10, tp90, color="red", alpha=0.06,
                    label=f"Threshold [{tp10:.2f}, {tp90:.2f}]")
    ax1.axhline(y=thr_mean, color="red", linestyle="--", linewidth=1.2,
                label=f"Threshold = {thr_mean:.2f}")
    ax1.axhline(y=initial_belief, color="gray", linestyle=":", linewidth=1,
                label=f"Prior = {initial_belief}")
    ax1.set(xlabel="Time Step", ylabel="$P(H_1)$",
            title="(a) Belief Evolution")
    ax1.legend(fontsize=FS-3, framealpha=0.9)

    final_beliefs = final_data["belief"].values
    ax2.hist(final_beliefs, bins=30, color="steelblue", edgecolor="white",
             linewidth=0.4, alpha=0.85)
    ax2.axvline(x=thr_mean, color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold = {thr_mean:.2f}")
    ax2.set(xlabel="$P(H_1)$", ylabel="Number of Agents",
            title=f"(b) Final Belief Distribution (Step {steps[-1]})")
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["belief"] = fig

    # --- Figure 4: Network + Similarity (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    # Network with flood count labels
    segs = []
    for u, v in model.G.edges():
        a_u, a_v = model.agents_by_node[u], model.agents_by_node[v]
        segs.append([(a_u.x, a_u.y), (a_v.x, a_v.y)])
    if segs:
        ax1.add_collection(LineCollection(
            segs, linewidths=0.3, colors="gray", alpha=0.2, zorder=1))
    adopted = [a for a in agents if a.is_retrofitted]
    not_adopted = [a for a in agents if not a.is_retrofitted]
    node_size = max(50, min(300, 10000 // max(len(agents), 1)))
    if not_adopted:
        ax1.scatter([a.x for a in not_adopted], [a.y for a in not_adopted],
                    c="lightgray", s=node_size, edgecolor="black",
                    linewidth=0.5, zorder=2)
        for a in not_adopted:
            ax1.text(a.x, a.y, str(a.flood_count), ha="center", va="center",
                     fontsize=max(7, min(10, 2200//max(len(agents),1))),
                     fontweight="bold", zorder=3)
    if adopted:
        sc = ax1.scatter([a.x for a in adopted], [a.y for a in adopted],
            c=[a.retrofit_step for a in adopted], cmap="YlGn", s=node_size,
            edgecolor="black", linewidth=0.5, zorder=2,
            vmin=1, vmax=model.current_step)
        plt.colorbar(sc, ax=ax1, shrink=0.6, pad=0.02).set_label(
            "Retrofit Step", fontsize=FS-2)
        for a in adopted:
            ax1.text(a.x, a.y, str(a.flood_count), ha="center", va="center",
                     fontsize=max(7, min(10, 2200//max(len(agents),1))),
                     fontweight="bold", zorder=3)
    ax1.set(xlabel="$x$", ylabel="$y$",
            title="(a) Social Network",
            xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax1.legend(handles=[
        Patch(facecolor="lightgray", edgecolor="black", label="Not Retrofitted"),
        Patch(facecolor="#31a354", edgecolor="black", label="Retrofitted")],
        loc="upper right", fontsize=FS-3, framealpha=0.9)

    # Similarity distribution
    sims = [d["similarity"] for _, _, d in model.G.edges(data=True)]
    ax2.hist(sims, bins=30, color="steelblue", edgecolor="white",
             linewidth=0.4, alpha=0.85)
    ax2.axvline(np.mean(sims), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean = {np.mean(sims):.3f}")
    ax2.set(xlabel="Jaccard Similarity $S(i,j)$", ylabel="Frequency",
            title="(b) Attribute Similarity Distribution")
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["network_sim"] = fig

    # --- Figure 5: Spatial map ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    segs2 = []
    for u, v in model.G.edges():
        a_u, a_v = model.agents_by_node[u], model.agents_by_node[v]
        segs2.append([(a_u.x, a_u.y), (a_v.x, a_v.y)])
    if segs2:
        ax.add_collection(LineCollection(
            segs2, linewidths=0.3, colors="gray", alpha=0.15, zorder=1))
    sc = ax.scatter([a.x for a in agents], [a.y for a in agents],
                    c=[a.z for a in agents], cmap="terrain", s=node_size,
                    alpha=0.8, edgecolor="black", linewidth=0.3, zorder=2)
    for a in agents:
        if a.is_retrofitted:
            ax.scatter(a.x, a.y, c="green", s=node_size, edgecolor="black",
                       linewidth=1.0, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02).set_label(
        "Elevation", fontsize=FS-2)
    ax.set(xlabel="$x$", ylabel="$y$",
           title="Elevation and Retrofit Status",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax.legend(handles=[
        Patch(facecolor="green", edgecolor="black", label="Retrofitted"),
        Patch(facecolor="lightgray", edgecolor="black", label="Not Retrofitted")],
        loc="upper right", fontsize=FS-3, framealpha=0.9)
    fig.tight_layout()
    figs["spatial"] = fig

    return figs


# ============================================================================
# TAB: RUN SIMULATION
# ============================================================================

with tab_sim:
    st.title("Run Simulation")

    # Parameter summary
    st.subheader("Current Configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", n_agents)
    c1.metric("Time Steps", time_steps)
    c2.metric("$\\lambda_{\\mathrm{flood}}$", f"{lambda_flood:.2f}")
    c2.metric("$\\lambda_{\\mathrm{social}}$", f"{lambda_social:.2f}")
    c3.metric("$\\lambda_{\\mathrm{similarity}}$", f"{lambda_similarity:.2f}")
    c3.metric("PMT Threshold", f"{pmt_mean:.2f}")
    c4.metric("Initial Belief", f"{initial_belief:.2f}")
    c4.metric("Mode", mode.replace(" Mode", ""))

    st.divider()

    if mode == "Case Study Mode" and uploaded_csv is None:
        st.warning("Upload a location CSV in the sidebar to use Case Study Mode.")

    # Run button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        apply_params()
        log_area = st.empty()
        progress_bar = st.progress(0)
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            log_area.code("\n".join(log_lines), language="text")

        log("Initializing model...")

        # Handle Case Study CSV
        if mode == "Case Study Mode" and uploaded_csv is not None:
            uploaded_csv.seek(0)
            csv_df = pd.read_csv(uploaded_csv)
            csv_path = "/tmp/_abm_upload.csv"
            csv_df.to_csv(csv_path, index=False)
            M.N_AGENTS = len(csv_df)
            M.SPATIAL_MODE = 0
            M.CSV_PATH = csv_path
            log(f"  Loaded {len(csv_df)} agents from uploaded CSV")

        model = M.FloodAdaptationModel()
        log(f"  Agents: {model.n_agents}")
        log(f"  Edges: {model.G.number_of_edges()}")
        log(f"  Neighborhoods: {model.n_neighborhoods} "
            f"(isolated: {int(np.sum(model.neighborhood_labels == -1))})")
        log(f"  Mean degree: {np.mean([d for _, d in model.G.degree()]):.1f}")
        progress_bar.progress(5)

        # Inject flood time series if provided
        if mode == "Case Study Mode" and uploaded_flood is not None:
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
            log(f"  Using uploaded flood time series ({len(flood_ts)} steps)")

        log("\nRunning simulation...")
        t0 = time.time()

        for i in range(model.time_steps):
            model.step()
            pct = int(5 + 85 * (i + 1) / model.time_steps)
            progress_bar.progress(pct)
            if (i + 1) % max(1, model.time_steps // 10) == 0:
                n_ret = sum(1 for a in model.agents if a.is_retrofitted)
                log(f"  Step {i+1:4d}: retrofitted = {n_ret}/{model.n_agents} "
                    f"({100*n_ret/model.n_agents:.1f}%)")

        elapsed = time.time() - t0
        agents = list(model.agents)
        n_ret = sum(1 for a in agents if a.is_retrofitted)

        log(f"\nSimulation complete in {elapsed:.1f}s")
        log(f"  Retrofitted: {n_ret}/{model.n_agents} "
            f"({100*n_ret/model.n_agents:.1f}%)")
        log(f"  Mean belief: {np.mean([a.belief for a in agents]):.4f}")
        log(f"  Mean flood count: {np.mean([a.flood_count for a in agents]):.1f}")

        progress_bar.progress(90, text="Generating figures...")
        log("\nGenerating figures...")

        st.session_state["model"] = model
        st.session_state["figs"] = make_figures(model)

        progress_bar.progress(100, text="Done.")
        log("Done. Go to the Results tab.")

        st.success(f"Retrofitted: {n_ret}/{model.n_agents} "
                   f"({100*n_ret/model.n_agents:.1f}%) in {elapsed:.1f}s")


# ============================================================================
# TAB: RESULTS
# ============================================================================

with tab_results:
    st.title("Results")

    if "figs" not in st.session_state:
        st.info("Run a simulation first to see results here.")
    else:
        figs = st.session_state["figs"]
        model = st.session_state["model"]

        st.header("Adoption and Flood History")
        st.pyplot(figs["adoption_flood"])

        st.header("Elevation and Observed Comparison")
        st.pyplot(figs["elev_comparison"])

        st.header("Belief Evolution")
        st.pyplot(figs["belief"])

        st.header("Network and Similarity")
        st.pyplot(figs["network_sim"])

        st.header("Spatial Map")
        col_sp, _ = st.columns([2, 1])
        with col_sp:
            st.pyplot(figs["spatial"])

        # Export
        st.divider()
        st.header("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            agent_csv = model.get_agent_dataframe().to_csv()
            st.download_button("Download Agent Data (CSV)",
                               agent_csv, "agent_data.csv", "text/csv",
                               use_container_width=True)
        with col2:
            model_csv = model.get_model_dataframe().to_csv()
            st.download_button("Download Model Data (CSV)",
                               model_csv, "model_data.csv", "text/csv",
                               use_container_width=True)
